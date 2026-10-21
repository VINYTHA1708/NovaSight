"""Script: evaluate the frozen primary artifact on test pools.

Reads the frozen artifact from artifacts/base/ and evaluates on:
  - test pools (known-class metrics, T1, T2, T3, T4)
  - unk_test (unknown-detection metrics)
  - enroll_pool (10 draws for T5/H1/H1-s)
  - new_test (post-enrollment metrics)

This script is READ-ONLY on the artifact. It never writes to artifacts/.
Results go to reports/results_{run_id}.json.

Usage:
    .venv\Scripts\python.exe scripts\evaluate.py --backbone B0 --feature L4 --held-out Sc --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging
from src.config import get_config, REPO_ROOT
from src.data.splits import load_manifest, KNOWN_PREFIXES, get_pool
from src.data.index import PREFIX_TO_CLASS
from src.models.gaussian import GaussianModel, fit_gaussian
from src.models.conformal import ConformalScores, build_conformal_scores, make_decision
from src.models.memory import NovaSightMemory, EnrolledClass, replay_events
from src.models.gate import run_gate
from scripts.cache_features import get_cache_path
from src.utils.hashing import sha256_file

logger = logging.getLogger(__name__)


def load_artifact(backbone_id: str, feature: str) -> tuple[GaussianModel, ConformalScores, dict]:
    """Load the frozen base artifact.

    Args:
        backbone_id: Backbone identifier.
        feature: Feature mode string.

    Returns:
        Tuple of (GaussianModel, ConformalScores, metadata dict).
    """
    artifact_dir = REPO_ROOT / "artifacts" / "base"
    meta_path = artifact_dir / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Load Gaussian
    g_data = np.load(artifact_dir / "gaussian.npz", allow_pickle=True)
    A = g_data["A"]
    class_names = [str(c) for c in g_data["class_names"]]
    n_c = list(g_data["n_c"])
    k = int(g_data["k"])
    feature_dim = int(g_data["feature_dim"])
    shrinkage_intensity = float(g_data["shrinkage_intensity"])
    condition_number = float(g_data["condition_number"])

    m_data = np.load(artifact_dir / "class_means.npz")
    class_means = [m_data[f"means_{i}"] for i in range(len(class_names))]
    n_sub = [list(m_data[f"n_sub_{i}"]) for i in range(len(class_names))]

    gaussian = GaussianModel(
        class_names=class_names,
        class_means=class_means,
        n_c=list(n_c),
        n_sub=[list(ns) for ns in n_sub],
        A=A,
        condition_number=condition_number,
        shrinkage_intensity=shrinkage_intensity,
        k=k,
        feature_dim=feature_dim,
    )

    # Load conformal
    c_data = np.load(artifact_dir / "conformal.npz")
    class_scores = {cn: c_data[f"scores_{cn}"] for cn in class_names}
    S_pool = c_data["S_pool"]
    tau_hat = float(c_data["tau_hat"][0])

    conformal = ConformalScores(
        class_scores=class_scores,
        S_pool=S_pool,
        tau_hat=tau_hat,
        alpha_default=meta.get("alpha_default", 0.05),
        alpha_grid=meta.get("alpha_grid", [0.01, 0.02, 0.05, 0.10, 0.20]),
    )
    return gaussian, conformal, meta


def load_features(backbone_id: str, feature: str, held_out: str, seed: int, pool: str, prefix: str) -> np.ndarray:
    """Load cached features for one class/pool.

    Args:
        backbone_id: Backbone identifier.
        feature: Feature mode.
        held_out: Held-out class prefix.
        seed: Split seed.
        pool: Pool name.
        prefix: Class prefix.

    Returns:
        (n, D) feature array.
    """
    path = get_cache_path(backbone_id, feature, held_out, seed, pool, prefix)
    if not path.exists():
        raise FileNotFoundError(f"Feature cache missing: {path}")
    return np.load(path)["features"]


def compute_auroc(scores_known: np.ndarray, scores_unknown: np.ndarray) -> float:
    """Compute AUROC. Lower score = more novel.

    Args:
        scores_known: Novelty scores for known class images (lower = more known).
        scores_unknown: Novelty scores for unknown class images.

    Returns:
        AUROC value (higher is better for known-class identification).
    """
    n_k = len(scores_known)
    n_u = len(scores_unknown)
    if n_k == 0 or n_u == 0:
        return float("nan")
    # Unknown should have HIGHER novelty score -> AUROC for detecting unknowns
    # by thresholding: unknown if score > threshold
    labels = np.concatenate([np.zeros(n_k), np.ones(n_u)])
    preds = np.concatenate([scores_known, scores_unknown])
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, preds))


def bootstrap_ci(metric_fn, n_bootstrap: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Placeholder for bootstrap CI computation.

    Args:
        metric_fn: Callable returning a scalar metric.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        (lower, upper) 95% CI.
    """
    rng = np.random.default_rng(seed)
    results = [metric_fn(rng) for _ in range(n_bootstrap)]
    return float(np.percentile(results, 2.5)), float(np.percentile(results, 97.5))


def main() -> None:
    """Run evaluation for one experiment run."""
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Evaluate primary artifact")
    parser.add_argument("--backbone", default="B0", choices=["B0", "B1", "B2"])
    parser.add_argument("--feature", default="L4")
    parser.add_argument("--held-out", default="Sc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    cfg = get_config()
    held_out = args.held_out
    split_seed = args.seed
    alpha = args.alpha
    run_id = f"H={held_out}_s={split_seed}"

    logger.info("Evaluating: run_id=%s, backbone=%s, feature=%s, alpha=%.2f",
                run_id, args.backbone, args.feature, alpha)
    t0 = time.time()

    gaussian, conformal, meta = load_artifact(args.backbone, args.feature)

    # ── T1: Known-class accuracy on TEST ─────────────────────────────────────
    logger.info("T1: Known-class test metrics...")
    known_prefixes = [p for p in KNOWN_PREFIXES if p != held_out]
    all_test_feats, all_test_labels, all_test_pred_labels = [], [], []
    per_class_results = {}

    for prefix in known_prefixes:
        cname = PREFIX_TO_CLASS[prefix]
        feats = load_features(args.backbone, args.feature, held_out, split_seed, "test", prefix)
        c_idx = gaussian.class_names.index(cname)
        for i in range(len(feats)):
            f = feats[i]
            scores = gaussian.all_inflated_scores(f)
            pred_idx = int(np.argmin(scores))
            all_test_feats.append(f)
            all_test_labels.append(c_idx)
            all_test_pred_labels.append(pred_idx)
        correct = sum(1 for i in range(len(feats))
                      if gaussian.class_names[int(np.argmin(gaussian.all_inflated_scores(feats[i])))] == cname)
        per_class_results[cname] = {"n": len(feats), "correct": correct,
                                    "accuracy": correct / len(feats)}

    all_test_feats = np.array(all_test_feats)
    all_test_labels = np.array(all_test_labels)
    all_test_pred_labels = np.array(all_test_pred_labels)
    overall_accuracy = float((all_test_labels == all_test_pred_labels).mean())
    logger.info("T1: Closed-set accuracy = %.4f", overall_accuracy)

    # ── T2/T3: AUROC, unknown rejection on TEST + UNK_TEST ──────────────────
    logger.info("T2/T3: Open-set metrics...")

    # Known novelty scores on test
    known_novelty_scores = gaussian.batch_novelty_scores(all_test_feats)

    # Unknown (held-out) novelty scores on unk_test
    unk_feats = load_features(args.backbone, args.feature, held_out, split_seed, "unk_test", held_out)
    unk_novelty_scores = gaussian.batch_novelty_scores(unk_feats)

    auroc = compute_auroc(known_novelty_scores, unk_novelty_scores)
    logger.info("T2: AUROC (NovaSight, min_c dt_c) = %.4f", auroc)

    # T3: rejection rates at alpha
    def get_verdict(f, alpha):
        scores = {c: gaussian.inflated_score(f, ci)
                  for ci, c in enumerate(gaussian.class_names)}
        pred_set, _ = conformal.prediction_set(scores, alpha)
        return make_decision(pred_set, gaussian.class_names[int(np.argmin(list(scores.values())))])

    known_rejected = sum(1 for f in all_test_feats
                         if get_verdict(f, alpha)["verdict"] == "unknown")
    unk_rejected = sum(1 for f in unk_feats
                       if get_verdict(f, alpha)["verdict"] == "unknown")
    known_frr = known_rejected / len(all_test_feats)
    unk_rejection_rate = unk_rejected / len(unk_feats)
    logger.info("T3 alpha=%.2f: known FRR=%.4f, unk rejection=%.4f",
                alpha, known_frr, unk_rejection_rate)

    # ── T5: Post-enrollment (10 draws from enroll_pool) ──────────────────────
    logger.info("T5: Post-enrollment evaluation (10 draws)...")
    enroll_feats = load_features(args.backbone, args.feature, held_out, split_seed, "enroll_pool", held_out)
    new_test_feats = load_features(args.backbone, args.feature, held_out, split_seed, "new_test", held_out)

    enroll_draws_per_run = cfg["experiments"]["enrollment_draws_per_run"]
    draw_seed = cfg["experiments"]["enrollment_draw_seed"] + split_seed
    rng = random.Random(draw_seed)

    enrollment_results = []
    for draw_idx in range(enroll_draws_per_run):
        draw_indices = rng.sample(range(len(enroll_feats)), 5)
        draw_feats = enroll_feats[draw_indices]
        draw_sha256s = [f"draw_{draw_idx}_img_{i}" for i in range(5)]

        # Run gate (always, record but never block per APPROVED-POLICY)
        gate_report = run_gate(
            embeddings=draw_feats,
            sha256s=draw_sha256s,
            class_name=f"{held_out}_enrolled",
            existing_class_names=gaussian.class_names,
            gaussian=gaussian,
            conformal=conformal,
            alpha_cons=cfg["gate"]["alpha_cons_grid"][1],  # default 0.01
            q_BH=cfg["gate"]["q_BH_grid"][0],             # default 0.05
            eps_max=cfg["gate"]["eps_max_grid"][1],        # default 0.05
        )

        # Enroll regardless of gate outcome [APPROVED-POLICY]
        import tempfile
        mem = NovaSightMemory(
            gaussian=gaussian, conformal=conformal,
            inheritance="H1",
            events_path=Path(tempfile.mktemp(suffix=".jsonl")),
        )
        cls = mem.enroll(
            name=f"{held_out}_enrolled",
            embeddings=draw_feats,
            sha256s=draw_sha256s,
            gate_report=gate_report.to_dict(),
        )

        # Evaluate on new_test (H1)
        h1_recalls = []
        h1s_scores_recalled = []
        for f in new_test_feats:
            score_h1 = cls.inflated_score(f, gaussian.A, conformal.tau_hat)
            p_h1 = conformal.p_value_from_pool(score_h1)
            h1_recalls.append(p_h1 > alpha)

        recall_h1 = float(np.mean(h1_recalls))

        enrollment_results.append({
            "draw_idx": draw_idx,
            "gate_outcome": gate_report.outcome,
            "recall_h1": recall_h1,
            "n_new_test": len(new_test_feats),
        })
        logger.info("Draw %d: gate=%s, recall_H1=%.4f",
                    draw_idx, gate_report.outcome, recall_h1)

    mean_recall_h1 = float(np.mean([r["recall_h1"] for r in enrollment_results]))
    logger.info("T5: Mean recall H1 = %.4f (over %d draws)", mean_recall_h1, enroll_draws_per_run)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "run_id": run_id,
        "backbone_id": args.backbone,
        "feature": args.feature,
        "held_out": held_out,
        "split_seed": split_seed,
        "alpha": alpha,
        "T1": {
            "overall_accuracy": overall_accuracy,
            "per_class": per_class_results,
        },
        "T2": {
            "AUROC_novasight": auroc,
            "novelty_score": "min_c_dt_c [APPROVED-POLICY]",
        },
        "T3": {
            "alpha": alpha,
            "known_FRR": known_frr,
            "unk_rejection_rate": unk_rejection_rate,
            "n_known_test": len(all_test_feats),
            "n_unk_test": len(unk_feats),
        },
        "T5": {
            "n_draws": enroll_draws_per_run,
            "mean_recall_H1": mean_recall_h1,
            "draws": enrollment_results,
        },
        "runtime_s": time.time() - t0,
    }

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    result_path = reports_dir / f"results_{run_id}_{args.backbone}_{args.feature}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    sha = sha256_file(result_path)
    logger.info("Results saved: %s (sha256: %s...)", result_path.name, sha[:16])

    # Print summary
    print(f"\n{'='*60}")
    print(f"RUN: {run_id}  backbone={args.backbone}  feature={args.feature}")
    print(f"{'='*60}")
    print(f"T1 Closed-set accuracy:    {overall_accuracy:.4f}")
    print(f"T2 AUROC (min_c dt_c):     {auroc:.4f}")
    print(f"T3 Known FRR (alpha={alpha}): {known_frr:.4f}")
    print(f"T3 Unk rejection rate:     {unk_rejection_rate:.4f}")
    print(f"T5 Mean recall H1:         {mean_recall_h1:.4f}")
    print(f"Runtime: {results['runtime_s']:.1f}s")


if __name__ == "__main__":
    main()

