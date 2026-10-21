"""Script: generate confusion matrix with 'unknown' column (M2 requirement).

Reads the frozen artifact and evaluates on test+unk_test pools.
Outputs reports/confusion_matrix_{run_id}.json and a markdown table.

Usage:
    .venv/Scripts/python.exe scripts/gen_confusion_matrix.py [--backbone B0] [--feature L4]
                                                              [--held-out Sc] [--seed 0] [--alpha 0.05]
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
from src.eval.metrics import collect_run_metrics, confusion_matrix_with_unknown
from scripts.evaluate import load_artifact, load_features

logger = logging.getLogger(__name__)


def main() -> None:
    """Generate confusion matrix with unknown column."""
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Generate confusion matrix (M2)")
    parser.add_argument("--backbone", default="B0", choices=["B0", "B1", "B2"])
    parser.add_argument("--feature", default="L4")
    parser.add_argument("--held-out", default="Sc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    run_id = f"H={args.held_out}_s={args.seed}"
    gaussian, conformal, meta = load_artifact(args.backbone, args.feature)

    known_prefixes = [p for p in KNOWN_PREFIXES if p != args.held_out]

    test_features = {}
    for prefix in known_prefixes:
        cname = PREFIX_TO_CLASS[prefix]
        test_features[cname] = load_features(args.backbone, args.feature, args.held_out, args.seed, "test", prefix)

    unk_features = load_features(args.backbone, args.feature, args.held_out, args.seed, "unk_test", args.held_out)

    metrics = collect_run_metrics(gaussian, conformal, test_features, unk_features, args.alpha)

    cm_result = metrics["confusion_matrix"]

    print("\n## Confusion Matrix (M2 requirement)")
    print(f"Run: {run_id}  backbone={args.backbone}  feature={args.feature}  alpha={args.alpha}")
    print(f"Held-out (unknown) class: {PREFIX_TO_CLASS.get(args.held_out, args.held_out)}\n")
    print(cm_result["markdown_table"])
    print(f"\nT1 accuracy:       {metrics['T1_accuracy']:.4f}")
    print(f"T2 AUROC:          {metrics['T2_AUROC']:.4f}")
    print(f"T2 FPR@95:         {metrics['T2_FPR95']:.4f}")
    print(f"T3 FRR:            {metrics['T3_FRR']:.4f}")
    print(f"T3 FDR:            {metrics['T3_FDR']:.4f}")
    print(f"T3 Unk rejection:  {metrics['T3_unk_rejection']:.4f}")

    # Save result
    out = REPO_ROOT / "reports" / f"confusion_matrix_{run_id}_{args.backbone}_{args.feature}.json"
    result_to_save = {
        "run_id": run_id,
        "backbone_id": args.backbone,
        "feature": args.feature,
        "held_out": args.held_out,
        "alpha": args.alpha,
        "T1_accuracy": metrics["T1_accuracy"],
        "T2_AUROC": metrics["T2_AUROC"],
        "T2_FPR95": metrics["T2_FPR95"],
        "T3_FRR": metrics["T3_FRR"],
        "T3_FDR": metrics["T3_FDR"],
        "T3_unk_rejection": metrics["T3_unk_rejection"],
        "confusion_matrix": cm_result["matrix"].tolist(),
        "row_names": cm_result["row_names"],
        "col_names": cm_result["col_names"],
        "markdown_table": cm_result["markdown_table"],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result_to_save, f, indent=2)
    logger.info("Confusion matrix saved to %s", out)


if __name__ == "__main__":
    main()
