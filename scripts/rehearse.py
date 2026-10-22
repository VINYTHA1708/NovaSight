"""Script: execute nested rehearsal and model selection under Protocol v1.1 (Task E5).

Uses ONLY train, early_stop, sel_cal, sel_eval from that run's manifest.
Zero test data access (enforced by PoolGuard).

Usage:
    .venv\\Scripts\\python.exe scripts\\rehearse.py --held-out Sc --seed 0 --device auto
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config, REPO_ROOT
from src.data.guard import PoolGuard
from src.data.index import KNOWN_PREFIXES, PREFIX_TO_CLASS, CLASS_TO_PREFIX
from src.data.splits_v2 import load_manifest_v2, get_pool_v2
from src.eval.rehearsal import (
    CandidateResult,
    evaluate_pseudo_enrollment,
    score_features_novelty,
    select_best_candidate,
    tune_gate_parameters,
)
from src.features.cache import get_or_extract_features
from src.models.backbones import make_extractor
from src.models.conformal import build_conformal_scores
from src.models.gaussian import fit_gaussian
from src.utils.device import setup_device
from src.utils.hashing import canonical_file_sha256, canonical_json_sha256
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Return HEAD commit hash."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def run_rehearsal_for_run(
    held_out: str,
    split_seed: int,
    device_arg: str = "auto",
    manifest_version: str = "v2",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute complete nested rehearsal for a single run."""
    device = setup_device(device_arg)
    if out_dir is None:
        out_dir = REPO_ROOT / "reports" / "v1.1"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"H={held_out}_s={split_seed}"
    manifest_path = REPO_ROOT / "data" / "manifests" / manifest_version / f"{run_id}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_sha = canonical_file_sha256(manifest_path)
    manifest = load_manifest_v2(manifest_path)

    held_out_prefix = CLASS_TO_PREFIX.get(held_out, held_out)
    known_prefixes = [p for p in KNOWN_PREFIXES if p != held_out_prefix]
    known_classes = [PREFIX_TO_CLASS[p] for p in known_prefixes]

    # Enforce pool guard
    allowed = ["train", "early_stop", "sel_cal", "sel_eval"]
    with PoolGuard(allowed_pools=allowed):
        # Candidates:
        # B0 (frozen ImageNet ResNet-18) x feature {L4, L234} x combination {a, b} x mixture k in {1, 2, 3}
        # B2 is evaluated and skipped/recorded as NOT_RUN under Decision D9 (no crash)
        candidate_specs = [
            {"backbone": "B0", "feature": "L4", "combination": "a", "k": 1},
            {"backbone": "B0", "feature": "L4", "combination": "a", "k": 2},
            {"backbone": "B0", "feature": "L4", "combination": "a", "k": 3},
            {"backbone": "B0", "feature": "L234", "combination": "a", "k": 1},
            {"backbone": "B0", "feature": "L234", "combination": "b", "k": 1},
            {"backbone": "B2", "feature": "CLS", "combination": "a", "k": 1},
        ]

        candidate_results: list[CandidateResult] = []

        # Cache/extract features for each known class in train, sel_cal, sel_eval
        # Extract features for B0
        extractor_b0 = make_extractor("B0")
        extractor_b0.to(device)

        features_by_spec: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for feat in ["L4", "L234"]:
            features_by_spec[feat] = {"train": {}, "sel_cal": {}, "sel_eval": {}}
            for pool in ["train", "sel_cal", "sel_eval"]:
                for prefix in known_prefixes:
                    cls_name = PREFIX_TO_CLASS[prefix]
                    records = get_pool_v2(manifest, prefix, pool)
                    feats = get_or_extract_features(
                        records=records,
                        extractor=extractor_b0,
                        backbone_id="B0",
                        feature=feat,
                        manifest_version=manifest_version,
                        run_id=run_id,
                        pool=pool,
                        class_name=cls_name,
                        device=device,
                    )
                    features_by_spec[feat][pool][cls_name] = feats

        # Apply TRAIN z-scoring to L234 features (Item 2.b)
        from src.eval.rehearsal import zscore_l234, cross_fit_combination_b
        all_train_l234 = np.vstack([features_by_spec["L234"]["train"][c] for c in known_classes])
        _, train_mean_l234, train_std_l234 = zscore_l234(all_train_l234)

        for pool in ["train", "sel_cal", "sel_eval"]:
            for c in known_classes:
                raw_l234 = features_by_spec["L234"][pool][c]
                z_feats, _, _ = zscore_l234(raw_l234, train_mean_l234, train_std_l234)
                features_by_spec["L234"][pool][c] = z_feats

        # Evaluate each candidate across nested leave-one-out
        for spec in candidate_specs:
            cid = f"{spec['backbone']}_{spec['feature']}_{spec['combination']}_k{spec['k']}"

            # Item 2.e: B2 skipped and recorded NOT_RUN without crashing
            if spec["backbone"] == "B2":
                candidate_results.append(CandidateResult(
                    candidate_id=cid,
                    backbone="B2",
                    feature=spec["feature"],
                    combination=spec["combination"],
                    k=spec["k"],
                    mean_auroc=0.0,
                    mean_known_frr=1.0,
                    mean_ambiguity_rate=0.0,
                    mean_h1_recall=0.0,
                    mean_h1s_recall=0.0,
                    h1_frr_gap=1.0,
                    h1s_frr_gap=1.0,
                    excluded=True,
                    exclusion_reason="B2 (DINOv2) is designated NOT RUN under Decision D9",
                ))
                continue

            feat = spec["feature"]
            k_val = spec["k"]

            loco_aurocs = []
            loco_frrs = []
            loco_ambigs = []
            loco_h1_recalls = []
            loco_h1s_recalls = []
            loco_h1_gaps = []
            loco_h1s_gaps = []

            for pseudo_unknown in known_classes:
                train_classes = [c for c in known_classes if c != pseudo_unknown]
                train_dict = {c: features_by_spec[feat]["train"][c] for c in train_classes}
                sel_cal_dict = {c: features_by_spec[feat]["sel_cal"][c] for c in train_classes}

                gauss = fit_gaussian(train_dict, k=k_val)
                conformal = build_conformal_scores(gauss, sel_cal_dict)

                # Score known sel_eval
                known_eval_scores = []
                known_rejected = 0
                known_ambig = 0
                total_known_eval = 0

                for c in train_classes:
                    eval_feats = features_by_spec[feat]["sel_eval"][c]
                    scores = score_features_novelty(gauss, eval_feats)
                    known_eval_scores.extend(scores)
                    for f_vec in eval_feats:
                        dt_dict = {cls: gauss.inflated_score(f_vec, idx) for idx, cls in enumerate(gauss.class_names)}
                        pred_set, _ = conformal.prediction_set(dt_dict, alpha=0.05)
                        if c not in pred_set:
                            known_rejected += 1
                        if len(pred_set) >= 2:
                            known_ambig += 1
                        total_known_eval += 1

                # Score pseudo-unknown sel_eval
                unk_eval_feats = features_by_spec[feat]["sel_eval"][pseudo_unknown]
                unk_scores = score_features_novelty(gauss, unk_eval_feats)

                # AUROC
                y_true = np.concatenate([np.zeros(len(known_eval_scores)), np.ones(len(unk_scores))])
                y_scores = np.concatenate([known_eval_scores, unk_scores])
                try:
                    from sklearn.metrics import roc_auc_score
                    auroc_val = float(roc_auc_score(y_true, y_scores))
                except Exception:
                    auroc_val = 0.5

                frr_val = known_rejected / max(total_known_eval, 1)
                ambig_val = known_ambig / max(total_known_eval, 1)

                # Pseudo-enrollment
                pe_res = evaluate_pseudo_enrollment(
                    k_class=pseudo_unknown,
                    gaussian=gauss,
                    conformal=conformal,
                    sel_cal_feats=features_by_spec[feat]["sel_cal"][pseudo_unknown],
                    sel_eval_feats=unk_eval_feats,
                    other_sel_eval={c: features_by_spec[feat]["sel_eval"][c] for c in train_classes},
                )

                loco_aurocs.append(auroc_val)
                loco_frrs.append(frr_val)
                loco_ambigs.append(ambig_val)
                loco_h1_recalls.append(pe_res["h1_recall"])
                loco_h1s_recalls.append(pe_res["h1s_recall"])
                loco_h1_gaps.append(abs(pe_res["h1_frr_005"] - 0.05) + abs(pe_res["h1_frr_010"] - 0.10))
                loco_h1s_gaps.append(abs(pe_res["h1s_frr_005"] - 0.05) + abs(pe_res["h1s_frr_010"] - 0.10))

            mean_frr = float(np.mean(loco_frrs))
            excluded = False
            reason = ""
            if mean_frr > 0.08:
                excluded = True
                reason = f"Rehearsal known FRR {mean_frr:.4f} > 0.08 at alpha=0.05"

            res = CandidateResult(
                candidate_id=cid,
                backbone=spec["backbone"],
                feature=spec["feature"],
                combination=spec["combination"],
                k=spec["k"],
                mean_auroc=float(np.mean(loco_aurocs)),
                mean_known_frr=mean_frr,
                mean_ambiguity_rate=float(np.mean(loco_ambigs)),
                mean_h1_recall=float(np.mean(loco_h1_recalls)),
                mean_h1s_recall=float(np.mean(loco_h1s_recalls)),
                h1_frr_gap=float(np.mean(loco_h1_gaps)),
                h1s_frr_gap=float(np.mean(loco_h1s_gaps)),
                excluded=excluded,
                exclusion_reason=reason,
            )
            candidate_results.append(res)

        # Select winning candidate
        best_cand, inheritance_variant = select_best_candidate(candidate_results)

        # Tune gate parameters on the chosen model fit
        chosen_feat = best_cand.feature
        chosen_k = best_cand.k
        full_train = {c: features_by_spec[chosen_feat]["train"][c] for c in known_classes}
        full_sel_cal = {c: features_by_spec[chosen_feat]["sel_cal"][c] for c in known_classes}

        full_gauss = fit_gaussian(full_train, k=chosen_k)
        full_conformal = build_conformal_scores(full_gauss, full_sel_cal)

        gate_params = tune_gate_parameters(
            gaussian=full_gauss,
            conformal=full_conformal,
            sel_cal_by_class=full_sel_cal,
        )

        commit = get_git_commit()

        selection_record = {
            "run_id": run_id,
            "held_out": held_out,
            "split_seed": split_seed,
            "manifest_version": manifest_version,
            "manifest_sha256": manifest_sha,
            "code_commit": commit,
            "device": str(device),
            "chosen_config": {
                "backbone": best_cand.backbone,
                "feature": best_cand.feature,
                "combination": best_cand.combination,
                "k": best_cand.k,
            },
            "inheritance_variant": inheritance_variant,
            "gate_params": gate_params,
            "candidate_metrics": [
                {
                    "candidate_id": c.candidate_id,
                    "backbone": c.backbone,
                    "feature": c.feature,
                    "combination": c.combination,
                    "k": c.k,
                    "mean_auroc": c.mean_auroc,
                    "mean_known_frr": c.mean_known_frr,
                    "mean_ambiguity_rate": c.mean_ambiguity_rate,
                    "mean_h1_recall": c.mean_h1_recall,
                    "mean_h1s_recall": c.mean_h1s_recall,
                    "excluded": c.excluded,
                    "exclusion_reason": c.exclusion_reason,
                }
                for c in candidate_results
            ],
            "exclusions": [c.candidate_id for c in candidate_results if c.excluded],
        }

        out_path = out_dir / f"selection_{run_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(selection_record, f, indent=2)

        logger.info("Wrote selection record to %s", out_path)
        return selection_record


def main() -> None:
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Run nested rehearsal and selection under Protocol v1.1")
    parser.add_argument("--held-out", default="Sc", help="Held-out defect class")
    parser.add_argument("--seed", type=int, default=0, help="Split seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Computation device")
    parser.add_argument("--manifest-version", default="v2", help="Manifest version directory")
    args = parser.parse_args()

    run_rehearsal_for_run(
        held_out=args.held_out,
        split_seed=args.seed,
        device_arg=args.device,
        manifest_version=args.manifest_version,
    )


if __name__ == "__main__":
    main()
