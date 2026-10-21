"""NovaSight memory and enrollment lifecycle (Part 4.7, 4.8, S4).

Memory is the deterministic replay of the event log over the base artifact.
Events are appended to state/events.jsonl (append-only).

Enrolled class state:
  - mu: mean embedding (D,)
  - n: number of examples used for the mean
  - embeddings: stored exemplar embeddings (5,)
  - sha256s: sha256 of each exemplar image
  - gate_report: gate outcome from enrollment
  - version: monotonically increasing per class
  - r_hat: scale ratio for H1-s

Score variants (Part 4.7):
  H1   : dt_new = d_new / 1.2  (plain inheritance, n=5 -> 1+1/5=1.2)
  H1-s : dt_new = d_new / (r_hat * tau_hat)  (scale-adaptive)

Graduation (Part 4.7): >= 19 confirmed production images -> own cal set.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_config, REPO_ROOT
from src.models.gaussian import GaussianModel
from src.models.conformal import ConformalScores

logger = logging.getLogger(__name__)

GRADUATION_THRESHOLD = 19   # confirmed production images before own cal set


@dataclass
class EnrolledClass:
    """State for one enrolled (new) class.

    Attributes:
        name: Class name as given by the operator.
        mu: Mean embedding (D,).
        n: Number of exemplars used (always 5 at enrollment).
        embeddings: List of 5 exemplar embeddings.
        sha256s: SHA-256 of each exemplar image file.
        gate_report: Gate outcome dict.
        inheritance: "H1" or "H1-s".
        r_hat: Scale ratio (for H1-s; None for H1).
        version: Monotonically increasing version counter.
        confirmed_count: Number of operator-confirmed production images.
        enrolled_at: ISO-8601 timestamp.
    """

    name: str
    mu: np.ndarray
    n: int
    embeddings: list[np.ndarray]
    sha256s: list[str]
    gate_report: dict[str, Any]
    inheritance: str   # "H1" or "H1-s"
    r_hat: float | None
    version: int = 0
    confirmed_count: int = 0
    enrolled_at: str = ""

    def inflated_score(
        self,
        f: np.ndarray,
        A: np.ndarray,
        tau_hat: float,
    ) -> float:
        """Compute the inflated score for a query feature.

        H1:   dt_new = d_new / (1 + 1/n)
        H1-s: score = (d_new / (1 + 1/n)) / r_hat

        Args:
            f: Query feature vector (D,).
            A: Shared precision matrix.
            tau_hat: Mean of S_pool (used by H1-s).

        Returns:
            Inflated score (to be compared against S_pool).
        """
        diff = f - self.mu
        d = float(diff @ A @ diff)
        dt = d / (1.0 + 1.0 / self.n)

        if self.inheritance == "H1-s" and self.r_hat is not None:
            return dt / self.r_hat
        return dt

    @property
    def has_graduated(self) -> bool:
        """True if this class has enough confirmed images to use its own cal set."""
        return self.confirmed_count >= GRADUATION_THRESHOLD


class NovaSightMemory:
    """In-memory state of known and enrolled classes.

    The state is derived from the base artifact (frozen) plus the event log.
    All writes go through append_event(); reads are always from in-memory state.

    Args:
        gaussian: Base GaussianModel (known classes only; immutable).
        conformal: Base ConformalScores (known classes; immutable).
        inheritance: Which variant to use for enrolled classes.
        events_path: Path to the append-only events JSONL file.
    """

    def __init__(
        self,
        gaussian: GaussianModel,
        conformal: ConformalScores,
        inheritance: str = "H1",
        events_path: Path | None = None,
    ) -> None:
        self.gaussian = gaussian
        self.conformal = conformal
        self.inheritance = inheritance
        self.enrolled: dict[str, EnrolledClass] = {}
        self._version: int = 0

        if events_path is None:
            cfg = get_config()
            events_path = REPO_ROOT / cfg["paths"]["state"] / "events.jsonl"
        self.events_path = events_path

    @property
    def all_class_names(self) -> list[str]:
        """All class names (known + enrolled)."""
        return self.gaussian.class_names + list(self.enrolled.keys())

    @property
    def known_class_names(self) -> list[str]:
        """Known (base) class names only."""
        return self.gaussian.class_names

    def novelty_score(self, f: np.ndarray) -> float:
        """Compute N(f) = min over ALL classes (known + enrolled).

        For known classes: uses gaussian inflated scores.
        For enrolled classes: uses the enrolled class variant (H1 or H1-s).

        Args:
            f: Feature vector (D,).

        Returns:
            Minimum inflated score.
        """
        scores = [self.gaussian.inflated_score(f, c)
                  for c in range(len(self.gaussian.class_names))]
        for enrolled_cls in self.enrolled.values():
            scores.append(
                enrolled_cls.inflated_score(f, self.gaussian.A, self.conformal.tau_hat)
            )
        return min(scores)

    def all_p_values(
        self,
        f: np.ndarray,
        alpha: float | None = None,
    ) -> dict[str, float]:
        """Compute p-values for all known classes.

        INVARIANT: only known (base) class p-values are computed.
        Enrolled classes use S_pool but their p-values DO NOT affect
        known-class p-values (S_c is immutable after calibration).

        Args:
            f: Feature vector (D,).
            alpha: Not used here; included for call-site consistency.

        Returns:
            Dict of known class_name -> p-value.
        """
        scores = {
            c: self.gaussian.inflated_score(f, c_idx)
            for c_idx, c in enumerate(self.gaussian.class_names)
        }
        return self.conformal.all_p_values(scores)

    def enrolled_p_value(self, f: np.ndarray, class_name: str) -> float:
        """Compute p-value for an enrolled class against S_pool.

        Args:
            f: Feature vector (D,).
            class_name: Name of the enrolled class.

        Returns:
            p-value from S_pool.
        """
        cls = self.enrolled[class_name]
        score = cls.inflated_score(f, self.gaussian.A, self.conformal.tau_hat)
        return self.conformal.p_value_from_pool(score)

    def full_prediction_set(
        self,
        f: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[set[str], dict[str, float]]:
        """Compute prediction set including enrolled classes.

        Known classes: use class-conditional S_c.
        Enrolled classes: use S_pool.

        Args:
            f: Feature vector.
            alpha: Significance level.

        Returns:
            Tuple of (prediction_set, all_p_values).
        """
        known_p_vals = self.all_p_values(f)
        all_p_vals = dict(known_p_vals)

        for name, cls in self.enrolled.items():
            score = cls.inflated_score(f, self.gaussian.A, self.conformal.tau_hat)
            p = self.conformal.p_value_from_pool(score)
            all_p_vals[name] = p

        pred_set = {c for c, p in all_p_vals.items() if p > alpha}
        return pred_set, all_p_vals

    def enroll(
        self,
        name: str,
        embeddings: np.ndarray,
        sha256s: list[str],
        gate_report: dict[str, Any],
        operator_id: str = "operator",
    ) -> EnrolledClass:
        """Add a new class to memory (post-gate).

        Computes mu_new and r_hat (for H1-s).

        Args:
            name: New class name.
            embeddings: (5, D) array of exemplar embeddings.
            sha256s: SHA-256 of each exemplar image.
            gate_report: Gate outcome dict.
            operator_id: Operator identifier for the event log.

        Returns:
            The enrolled class object.
        """
        mu_new = embeddings.mean(axis=0)

        # Compute r_hat for H1-s: mean of 5 leave-one-out distances (inflated by 1+1/4)
        loo_scores = []
        for i in range(5):
            mu_others = (embeddings.sum(axis=0) - embeddings[i]) / 4.0
            diff = embeddings[i] - mu_others
            d = float(diff @ self.gaussian.A @ diff)
            dt = d / (1.0 + 1.0 / 4)
            loo_scores.append(dt)
        r_hat = float(np.mean(loo_scores)) / self.conformal.tau_hat if self.conformal.tau_hat > 0 else 1.0

        cls = EnrolledClass(
            name=name,
            mu=mu_new,
            n=5,
            embeddings=list(embeddings),
            sha256s=sha256s,
            gate_report=gate_report,
            inheritance=self.inheritance,
            r_hat=r_hat,
            version=0,
            confirmed_count=0,
            enrolled_at=datetime.now(timezone.utc).isoformat(),
        )
        self.enrolled[name] = cls
        self._append_event({
            "type": "enroll",
            "class_name": name,
            "n": 5,
            "mu": mu_new.tolist(),
            "embeddings": [e.tolist() for e in embeddings],
            "sha256s": sha256s,
            "r_hat": r_hat,
            "gate_report": gate_report,
            "operator_id": operator_id,
        })
        logger.info("Enrolled class '%s' (r_hat=%.4f)", name, r_hat)
        return cls

    def remove(self, name: str, operator_id: str = "operator") -> None:
        """Remove an enrolled class from memory.

        Base classes are immutable and cannot be removed.

        Args:
            name: Class name to remove.
            operator_id: Operator identifier.

        Raises:
            ValueError: If name is a base class or not enrolled.
        """
        if name in self.gaussian.class_names:
            raise ValueError(f"Cannot remove base class '{name}' from the UI.")
        if name not in self.enrolled:
            raise ValueError(f"Class '{name}' is not enrolled.")
        del self.enrolled[name]
        self._append_event({"type": "remove", "class_name": name, "operator_id": operator_id})
        logger.info("Removed enrolled class '%s'", name)

    def rename(self, old_name: str, new_name: str, operator_id: str = "operator") -> None:
        """Rename an enrolled class.

        Args:
            old_name: Current class name.
            new_name: New class name.
            operator_id: Operator identifier.
        """
        if old_name in self.gaussian.class_names:
            raise ValueError("Cannot rename base classes.")
        if old_name not in self.enrolled:
            raise ValueError(f"Class '{old_name}' is not enrolled.")
        if new_name in self.all_class_names:
            raise ValueError(f"Name '{new_name}' already exists.")
        cls = self.enrolled.pop(old_name)
        cls = EnrolledClass(**{**asdict(cls), "name": new_name,
                               "mu": cls.mu, "embeddings": cls.embeddings})
        self.enrolled[new_name] = cls
        self._append_event({
            "type": "rename",
            "old_name": old_name,
            "new_name": new_name,
            "operator_id": operator_id,
        })

    def merge(self, name_a: str, name_b: str, merged_name: str, operator_id: str = "operator") -> None:
        """Merge two enrolled classes.

        Pools the stored embeddings (10 total), recomputes mean with n=10.

        Args:
            name_a: First class name.
            name_b: Second class name.
            merged_name: Name for the merged class.
            operator_id: Operator identifier.
        """
        if name_a not in self.enrolled or name_b not in self.enrolled:
            raise ValueError(f"Both '{name_a}' and '{name_b}' must be enrolled.")
        cls_a = self.enrolled[name_a]
        cls_b = self.enrolled[name_b]
        all_embs = cls_a.embeddings + cls_b.embeddings   # 10
        all_sha256s = cls_a.sha256s + cls_b.sha256s
        embs_arr = np.array(all_embs)
        mu_merged = embs_arr.mean(axis=0)
        n_merged = len(all_embs)

        # Recompute r_hat with merged embeddings (use leave-one-out approach)
        loo_scores = []
        A = self.gaussian.A
        for i in range(n_merged):
            others = np.array([e for j, e in enumerate(all_embs) if j != i])
            mu_others = others.mean(axis=0)
            diff = all_embs[i] - mu_others
            d = float(diff @ A @ diff)
            dt = d / (1.0 + 1.0 / (n_merged - 1))
            loo_scores.append(dt)
        r_hat = float(np.mean(loo_scores)) / self.conformal.tau_hat if self.conformal.tau_hat > 0 else 1.0

        merged_cls = EnrolledClass(
            name=merged_name,
            mu=mu_merged,
            n=n_merged,
            embeddings=all_embs,
            sha256s=all_sha256s,
            gate_report={"merged_from": [name_a, name_b]},
            inheritance=self.inheritance,
            r_hat=r_hat,
            version=0,
            confirmed_count=0,
            enrolled_at=datetime.now(timezone.utc).isoformat(),
        )
        del self.enrolled[name_a]
        del self.enrolled[name_b]
        self.enrolled[merged_name] = merged_cls
        self._append_event({
            "type": "merge",
            "name_a": name_a,
            "name_b": name_b,
            "merged_name": merged_name,
            "n": n_merged,
            "operator_id": operator_id,
        })
        logger.info("Merged '%s' + '%s' -> '%s' (n=%d)", name_a, name_b, merged_name, n_merged)

    def get_version_hash(self) -> str:
        """Compute a content hash of the current memory state.

        Returns:
            SHA-256 hex string representing current enrolled state.
        """
        state_str = json.dumps(
            {name: cls.sha256s for name, cls in sorted(self.enrolled.items())},
            sort_keys=True,
        )
        return hashlib.sha256(state_str.encode()).hexdigest()

    def _append_event(self, event_data: dict[str, Any]) -> None:
        """Append an event to the events.jsonl log.

        Args:
            event_data: Event payload dict.
        """
        event: dict[str, Any] = {
            "uuid": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": self._version,
            **event_data,
        }
        self._version += 1
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=_numpy_default) + "\n")

    def export_state(self) -> dict[str, Any]:
        """Export the current state as a JSON-serialisable dict.

        Returns:
            State dict including all enrolled classes.
        """
        return {
            "version": self._version,
            "version_hash": self.get_version_hash(),
            "known_classes": self.gaussian.class_names,
            "enrolled_classes": [
                {
                    "name": cls.name,
                    "n": cls.n,
                    "sha256s": cls.sha256s,
                    "gate_report": cls.gate_report,
                    "inheritance": cls.inheritance,
                    "r_hat": cls.r_hat,
                    "version": cls.version,
                    "confirmed_count": cls.confirmed_count,
                    "enrolled_at": cls.enrolled_at,
                    "graduated": cls.has_graduated,
                }
                for cls in self.enrolled.values()
            ],
        }


def _numpy_default(obj):
    """JSON serialisation helper for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


def replay_events(
    events_path: Path,
    gaussian: GaussianModel,
    conformal: ConformalScores,
    inheritance: str = "H1",
    up_to_version: int | None = None,
) -> NovaSightMemory:
    """Replay the event log to reconstruct memory state.

    Args:
        events_path: Path to events.jsonl.
        gaussian: Base GaussianModel.
        conformal: Base ConformalScores.
        inheritance: Which variant to use.
        up_to_version: If given, replay only events with version < this value.

    Returns:
        Reconstructed NovaSightMemory.
    """
    memory = NovaSightMemory(
        gaussian=gaussian,
        conformal=conformal,
        inheritance=inheritance,
        events_path=events_path,
    )
    if not events_path.exists():
        return memory

    with open(events_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            if up_to_version is not None and event.get("version", 0) >= up_to_version:
                break
            _apply_event(memory, event)
    return memory


def _apply_event(memory: NovaSightMemory, event: dict[str, Any]) -> None:
    """Apply a single event to an in-memory state.

    Args:
        memory: NovaSightMemory to modify.
        event: Event dict from the log.
    """
    etype = event["type"]
    if etype == "enroll":
        embeddings = np.array(event["embeddings"])
        mu = np.array(event["mu"])
        cls = EnrolledClass(
            name=event["class_name"],
            mu=mu,
            n=event["n"],
            embeddings=list(embeddings),
            sha256s=event["sha256s"],
            gate_report=event.get("gate_report", {}),
            inheritance=memory.inheritance,
            r_hat=event.get("r_hat"),
            version=event.get("version", 0),
            enrolled_at=event.get("timestamp", ""),
        )
        memory.enrolled[cls.name] = cls
    elif etype == "remove":
        memory.enrolled.pop(event["class_name"], None)
    elif etype == "rename":
        cls = memory.enrolled.pop(event["old_name"], None)
        if cls is not None:
            cls = EnrolledClass(**{**asdict(cls), "name": event["new_name"],
                                   "mu": cls.mu, "embeddings": cls.embeddings})
            memory.enrolled[event["new_name"]] = cls
    elif etype == "merge":
        cls_a = memory.enrolled.pop(event["name_a"], None)
        cls_b = memory.enrolled.pop(event["name_b"], None)
        if cls_a and cls_b:
            all_embs = cls_a.embeddings + cls_b.embeddings
            embs_arr = np.array(all_embs)
            mu_merged = embs_arr.mean(axis=0)
            merged_cls = EnrolledClass(
                name=event["merged_name"],
                mu=mu_merged,
                n=event.get("n", len(all_embs)),
                embeddings=all_embs,
                sha256s=cls_a.sha256s + cls_b.sha256s,
                gate_report={"merged_from": [event["name_a"], event["name_b"]]},
                inheritance=memory.inheritance,
                r_hat=None,
                version=event.get("version", 0),
                enrolled_at=event.get("timestamp", ""),
            )
            memory.enrolled[event["merged_name"]] = merged_cls
    memory._version = event.get("version", memory._version) + 1
