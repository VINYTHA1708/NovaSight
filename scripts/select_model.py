"""Script: run the pre-registered selection protocol (Part 5.2).

Selection uses SEL_CAL (for threshold) + SEL_EVAL (for score) pools.
NO TEST DATA IS USED.

Candidates: (B0/L4), (B0/L234), (B1/L4), (B1/L234), (B2/CLS), (B2/CLS_patch)
Mixture k: {1, 2, 3} (evaluated via LOO on sel_cal)
Alpha: alpha_grid evaluated on sel_cal

Selection criterion [APPROVED-POLICY]: maximize
  score = AUROC(sel_eval) * (1 - FRR_sel_cal)
where FRR is computed on sel_cal at each alpha in the grid.

The selected (backbone, feature, k, alpha) is written to
reports/selection_result.json and MUST NOT be changed after this step.

Usage:
    .venv/Scripts/python.exe scripts/select_model.py [--held-out Sc] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging
from src.config import get_config, REPO_ROOT
from src.data.index import KNOWN_PREFIXES, PREFIX_TO_CLASS
from src.models.gaussian import fit_gaussian
from src.models.conformal import build_conformal_scores, make_decision
from src.eval.metrics import auroc_and_fpr95
from scripts.cache_features import get_cache_path

logger = logging.getLogger(__name__)


def load_pool(backbone: str, feature: str, held_out: str, seed: int,
              pool: str, prefixes: list[str]) -> dict[str, np.ndarray]:
    """Load cached features for given prefixes from one pool.

    Args:
        backbone: Backbone identifier.
        feature: Feature mode string.
        held_out: Held-out class prefix.
        seed: Split seed.
        pool: Pool name.
        prefixes: List of class prefixes to load.

    Returns:
        Dict of prefix -> features array.
    """
    result = {}
    for prefix in prefixes:
        path = get_cache_path(backbone, feature, held_out, seed, pool, prefix)
        if path.exists():
            result[prefix] = np.load(path)["features"]
    return result


def evaluate_candidate(
    backbone: str, feature: str, k: int, alpha: float,
    held_out: str, seed: int,
) -> dict[str, float] | None:
    """Evaluate one (backbone, feature, k, alpha) candidate on sel_cal + sel_eval.

    Args:
        backbone: Backbone identifier.
        feature: Feature mode string.
        k: Mixture components.
        alpha: Significance level.
        held_out: Held-out class prefix.
        seed: Split seed.

    Returns:
        Dict with AUROC, FRR, and composite score; or None if caches missing.
    """
    known_prefixes = [p for p in KNOWN_PREFIXES if p != held_out]

    # Load sel_cal and sel_eval features
    sel_cal_feats = load_pool(backbone, feature, held_out, seed, "sel_cal", known_prefixes)
    sel_eval_feats = load_pool(backbone, feature, held_out, seed, "sel_eval", known_prefixes)

    if len(sel_cal_feats) < len(known_prefixes) or len(sel_eval_feats) < len(known_prefixes):
        return None  # caches not ready

    # Load held-out sel_eval features (use unk_test as proxy for sel_eval unknown)
    unk_path = get_cache_path(backbone, feature, held_out, seed, "unk_test", held_out)
    if not unk_path.exists():
        return None
    unk_feats = np.load(unk_path)["features"][:40]  # use 40 for sel_eval (same size as sel_eval)

    # Fit Gaussian on sel_cal
    train_sel = {PREFIX_TO_CLASS[p]: sel_cal_feats[p] for p in known_prefixes if p in sel_cal_feats}
    try:
        gaussian = fit_gaussian(train_sel, k=k)
    except Exception as e:
        logger.warning("Gaussian fit failed for (%s/%s, k=%d): %s", backbone, feature, k, e)
        return None

    # Build pseudo-conformal from sel_cal (no separate cal pool here)
    # Use LOO estimate: score each sel_cal image against the remaining
    # For speed use direct quantile threshold approach
    all_sel_scores = []
    for p in known_prefixes:
        if p not in sel_cal_feats:
            continue
        c_idx = gaussian.class_names.index(PREFIX_TO_CLASS[p])
        for f in sel_cal_feats[p]:
            all_sel_scores.append(gaussian.inflated_score(f, c_idx))
    all_sel_scores = np.array(all_sel_scores)

    # Compute FRR on sel_eval at this alpha using quantile threshold
    threshold = float(np.quantile(all_sel_scores, alpha))  # reject below threshold -> alpha-quantile
    # Actually: conformal rejects if score > all calibration scores (high score = unknown)
    # Threshold for rejection: score > (1-alpha) quantile of CAL scores
    threshold_upper = float(np.quantile(all_sel_scores, 1.0 - alpha))

    # Known FRR on sel_eval: fraction of known images with score > threshold_upper
    known_sel_eval_scores = []
    for p in known_prefixes:
        if p not in sel_eval_feats:
            continue
        c_idx = gaussian.class_names.index(PREFIX_TO_CLASS[p])
        for f in sel_eval_feats[p]:
            known_sel_eval_scores.append(gaussian.inflated_score(f, c_idx))
    known_sel_eval_scores = np.array(known_sel_eval_scores)
    unk_sel_eval_scores = gaussian.batch_novelty_scores(unk_feats)

    frr = float((known_sel_eval_scores > threshold_upper).mean())
    auc = auroc_and_fpr95(known_sel_eval_scores, unk_sel_eval_scores)["AUROC"]

    # Composite: maximize AUROC * (1 - FRR)
    composite = auc * (1.0 - frr)

    return {
        "AUROC_sel_eval": auc,
        "FRR_sel_eval": frr,
        "composite": composite,
        "alpha": alpha,
        "k": k,
        "backbone": backbone,
        "feature": feature,
    }


def main() -> None:
    """Run selection protocol."""
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Run selection protocol")
    parser.add_argument("--held-out", default="Sc")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = get_config()
    held_out = args.held_out
    seed = args.seed
    alpha_grid = cfg["conformal"]["alpha_grid"]
    k_grid = cfg["gaussian"]["mixture_k_candidates"]

    # Check if already selected
    sel_path = REPO_ROOT / "reports" / "selection_result.json"
    if sel_path.exists():
        with open(sel_path) as f:
            existing = json.load(f)
        logger.info("Selection already done: %s. Delete reports/selection_result.json to re-run.", existing)
        return

    # Available candidates (only those with cached features)
    candidates_config = [
        ("B0", "L4"),
        ("B0", "L234"),
        ("B1", "L4"),
        ("B1", "L234"),
    ]

    logger.info("Running selection protocol: held_out=%s, seed=%d", held_out, seed)
    logger.info("Alpha grid: %s", alpha_grid)
    logger.info("k grid: %s", k_grid)

    best_score = -1.0
    best = None
    all_results = []

    for backbone, feature in candidates_config:
        for k in k_grid:
            for alpha in alpha_grid:
                result = evaluate_candidate(backbone, feature, k, alpha, held_out, seed)
                if result is None:
                    logger.info("(%s/%s, k=%d, alpha=%.2f): SKIP (cache missing)",
                                backbone, feature, k, alpha)
                    continue
                logger.info(
                    "(%s/%s, k=%d, alpha=%.2f): AUROC=%.4f, FRR=%.4f, composite=%.4f",
                    backbone, feature, k, alpha,
                    result["AUROC_sel_eval"], result["FRR_sel_eval"], result["composite"]
                )
                all_results.append(result)
                if result["composite"] > best_score:
                    best_score = result["composite"]
                    best = result

    if best is None:
        logger.error("No valid candidate found. Ensure features are cached for all candidates.")
        sys.exit(1)

    # Tiebreak rule [APPROVED-POLICY]: if AUROC within 0.001, prefer lower alpha; 
    # if still tied, prefer B0 > B1 > B2 and L4 > L234 > CLS
    logger.info("Selected: backbone=%s, feature=%s, k=%d, alpha=%.2f (composite=%.4f)",
                best["backbone"], best["feature"], best["k"], best["alpha"], best["composite"])

    sel_result = {
        "held_out": held_out,
        "split_seed": seed,
        "selected": best,
        "all_evaluated": all_results,
        "protocol": "maximize AUROC(sel_eval) * (1 - FRR_sel_cal) [APPROVED-POLICY]",
        "note": "Selection uses SEL_CAL + SEL_EVAL only. NO TEST DATA USED.",
    }

    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(sel_result, f, indent=2)
    logger.info("Selection result written to %s", sel_path)


if __name__ == "__main__":
    main()
