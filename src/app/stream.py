"""Simulated manufacturing production line stream (Stage S6).

Runs background thread simulating high-speed steel surface defect inspection:
  - Line speed: 0.5 to 5.0 images/sec (default 1.0)
  - Allowed image mix:
      85% known-class images (from sel_eval pool)
      15% novel defect images (from held_enroll_pool Sc)
      Strictly zero reads from test, unk_test, new_test, or cal pools.
  - Processing:
      Preprocess -> ONNX ResNet18 -> RuntimeScorer + Mondrian Conformal
  - Records inspections to database and maintains recent ticker for UI.
  - Supports burst injection (5 Sc images in rapid succession).
"""
from __future__ import annotations

import collections
import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from src.app.storage import StorageBackend
from src.config import REPO_ROOT
from src.runtime.artifact import load_runtime_artifact
from src.runtime.memory import RuntimeMemory
from src.runtime.onnx_extractor import OnnxFeatureExtractor
from src.runtime.preprocess import preprocess_image, load_and_check_app_image_numpy
from src.runtime.scoring import RuntimeScorer, make_decision
from src.utils.hashing import sha256_bytes

logger = logging.getLogger(__name__)


class ProductionStream:
    """Background simulator for manufacturing production line inspection."""

    def __init__(
        self,
        storage: StorageBackend,
        scorer: RuntimeScorer,
        extractor: OnnxFeatureExtractor,
        get_memory_fn: Callable[[], RuntimeMemory],
        speed_fps: float = 1.0,
        alpha: float = 0.05,
    ) -> None:
        self.storage = storage
        self.scorer = scorer
        self.extractor = extractor
        self.get_memory_fn = get_memory_fn
        self.speed_fps = max(0.5, min(5.0, speed_fps))
        self.alpha = alpha

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()   # Not paused initially
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Image pools (strictly sel_eval and held_enroll_pool)
        self.known_pool_paths: list[Path] = []
        self.novel_pool_paths: list[Path] = []
        self._load_image_pools()

        # Burst queue for defect injections
        self._burst_queue: collections.deque[Path] = collections.deque()

        # Rolling recent inspections (last 20)
        self.recent_inspections: collections.deque[dict[str, Any]] = collections.deque(maxlen=20)
        self.total_stream_inspections = 0
        self.flagged_stream_inspections = 0

    def _load_image_pools(self) -> None:
        """Load paths from H=Sc_s=0.json strictly for sel_eval and held_enroll_pool."""
        manifest_path = REPO_ROOT / "data" / "manifests" / "H=Sc_s=0.json"
        if not manifest_path.exists():
            logger.warning("Manifest not found: %s", manifest_path)
            return

        with open(manifest_path, encoding="utf-8") as f:
            mf = json.load(f)

        # 1. Known classes sel_eval
        for c in ["Cr", "In", "Pa", "PS", "RS"]:
            pool_key = f"{c}_sel_eval"
            for item in mf["pools"].get(pool_key, []):
                p = REPO_ROOT / item["path"]
                if p.exists():
                    self.known_pool_paths.append(p)

        # 2. Novel class Sc held_enroll_pool
        for item in mf["pools"].get("held_enroll_pool", []):
            p = REPO_ROOT / item["path"]
            if p.exists():
                self.novel_pool_paths.append(p)

        logger.info(
            "ProductionStream loaded %d known images and %d novel images.",
            len(self.known_pool_paths),
            len(self.novel_pool_paths),
        )

    @property
    def is_running(self) -> bool:
        """True if background thread is active and not stopped."""
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        """True if stream is currently paused."""
        return not self._pause_event.is_set()

    def set_speed(self, fps: float) -> None:
        """Adjust stream inspection rate (0.5 to 5.0 images/sec)."""
        with self._lock:
            self.speed_fps = max(0.5, min(5.0, float(fps)))

    def start(self) -> None:
        """Start or restart the background stream."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._pause_event.set()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ProductionStreamThread")
        self._thread.start()
        logger.info("ProductionStream started at %.1f img/sec", self.speed_fps)

    def pause(self) -> None:
        """Pause inspection processing."""
        self._pause_event.clear()
        logger.info("ProductionStream paused.")

    def resume(self) -> None:
        """Resume inspection processing."""
        self._pause_event.set()
        logger.info("ProductionStream resumed.")

    def stop(self) -> None:
        """Stop background stream and join thread."""
        self._stop_event.set()
        self._pause_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        logger.info("ProductionStream stopped.")

    def inject_defect_burst(self, count: int = 5) -> int:
        """Inject novel Sc defect images into urgent queue."""
        if not self.novel_pool_paths:
            return 0
        samples = random.sample(self.novel_pool_paths, min(count, len(self.novel_pool_paths)))
        with self._lock:
            for s in samples:
                self._burst_queue.append(s)
        logger.info("Injected %d defect images into burst queue.", len(samples))
        return len(samples)

    def process_one_image(self, img_path: Path | None = None) -> dict[str, Any]:
        """Synchronously process a single image and record to DB."""
        if img_path is None:
            # Check burst queue first
            with self._lock:
                if self._burst_queue:
                    img_path = self._burst_queue.popleft()
                elif self.novel_pool_paths and random.random() < 0.15:
                    img_path = random.choice(self.novel_pool_paths)
                elif self.known_pool_paths:
                    img_path = random.choice(self.known_pool_paths)
                else:
                    return {}

        t0 = time.perf_counter()
        pil_img = Image.open(img_path).convert("RGB")
        img_gray = load_and_check_app_image_numpy(pil_img)
        inp_tensor = preprocess_image(img_gray)
        feat = self.extractor(inp_tensor)

        mem = self.get_memory_fn()
        pred_set, all_p_vals = mem.full_prediction_set(feat, self.alpha)

        scores = {c: self.scorer.inflated_score(feat, ci) for ci, c in enumerate(self.scorer.class_names)}
        for name, cls in mem.enrolled.items():
            scores[name] = cls.inflated_score(feat, self.scorer.A, self.scorer.artifact.tau_hat)

        c_hat = min(scores.keys(), key=lambda c: scores[c])
        decision = make_decision(pred_set, c_hat, list(mem.enrolled.keys()))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_bytes = pil_img.tobytes()
        sha256 = sha256_bytes(raw_bytes)
        flag_op = decision["verdict"] in ("unknown", "ambiguous")

        record = {
            "image_sha256": sha256,
            "image_filename": img_path.name,
            "verdict": decision["verdict"],
            "class_name": decision["class_name"] or c_hat,
            "novelty_score": float(mem.novelty_score(feat)),
            "prediction_set": sorted(pred_set),
            "details": {
                "closed_set_label": c_hat,
                "novelty_score": float(mem.novelty_score(feat)),
                "per_class_dt": {k: float(v) for k, v in scores.items()},
                "p_values": {k: float(v) for k, v in all_p_vals.items()},
                "prediction_set": sorted(pred_set),
                "alpha": self.alpha,
                "source_file": img_path.name,
            },
            "flagged_for_op": flag_op,
            "latency_ms": latency_ms,
        }

        db_id = self.storage.record_inspection(record)
        record["id"] = db_id
        record["timestamp"] = time.strftime("%H:%M:%S")

        with self._lock:
            self.recent_inspections.append(record)
            self.total_stream_inspections += 1
            if flag_op:
                self.flagged_stream_inspections += 1

        return record

    def _run_loop(self) -> None:
        """Worker loop for production stream thread."""
        while not self._stop_event.is_set():
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            try:
                self.process_one_image()
            except Exception as exc:
                logger.error("Error processing stream image: %s", exc)

            with self._lock:
                delay = 1.0 / self.speed_fps if self.speed_fps > 0 else 1.0
            time.sleep(delay)

    def get_recent_inspections(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return snapshot of most recent stream inspections."""
        with self._lock:
            return list(self.recent_inspections)[-limit:]
