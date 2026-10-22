"""Evaluator for NovaSight Protocol v1.1 (Task E7).

Reads the frozen artifact from artifacts/runs/{protocol}/{run_id}/ (READ-ONLY).
Evaluates on test, unk_test, enroll_pool, new_test.
Produces Tables T1-T5, OSCR curve, Risk-Coverage curve, and confusion matrix.
Outputs to reports/{protocol}/results_{run_id}.json.

Usage:
    .venv\\Scripts\\python.exe scripts\\evaluate_v11.py --held-out Sc --seed 0 --device auto
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT
from src.data.guard import PoolGuard
from src.data.index import KNOWN_PREFIXES, PREFIX_TO_CLASS, CLASS_TO_PREFIX
from src.data.splits_v2 import load_manifest_v2, get_pool_v2
from src.eval.metrics_v11 import (
    AMBIGUOUS_LABEL,
    UNKNOWN_LABEL,
    aupr_metric,
    auroc_metric,
    compute_beta_spread,
    compute_bootstrap_ci,
    compute_decision_rates,
    confusion_matrix_v11,
    fpr95_metric,
    paired_mcnemar_test,
)
from src.features.cache import get_or_extract_features
from src.models.backbones import make_extractor
from src.models.conformal import ConformalScores
from src.models.gate import run_gate
from src.models.gaussian import GaussianModel
from src.utils.device import get_device_info, setup_device
from src.utils.hashing import canonical_file_sha256
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_artifact(artifact_dir: Path) -> tuple[GaussianModel, ConformalScores, dict[str, Any]]:
    """Load Gaussian model and conformal scores from frozen artifact."""
    with open(artifact_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    # Gaussian
    g_data = np.load(artifact_dir / "gaussian.npz")
    A = g_data["A"]
    class_names = [str(c) for c in g_data["class_names"]]
    n_c = list(g_data["n_c"])
    k = int(g_data["k"][0])
    feature_dim = int(g_data["feature_dim"][0])
    shrinkage = float(g_data["shrinkage_intensity"][0])
    cond_num = float(g_data["condition_number"][0])

    # Means
    m_data = np.load(artifact_dir / "class_means.npz")
    class_means = [m_data[f"means_{i}"] for i in range(len(class_names))]
    n_sub = [list(m_data[f"n_sub_{i}"]) for i in range(len(class_names))]

    gaussian = GaussianModel(
        class_names=class_names,
        class_means=class_means,
        n_c=n_c,
        n_sub=n_sub,
        A=A,
        condition_number=cond_num,
        shrinkage_intensity=shrinkage,
        k=k,
        feature_dim=feature_dim,
    )

    # Conformal
    c_data = np.load(artifact_dir / "conformal.npz")
    class_scores = {c: c_data[f"scores_{c}"] for c in class_names}
    s_pool = c_data["S_pool"]
    tau_hat = float(c_data["tau_hat"][0])

    conformal = ConformalScores(
        class_scores=class_scores,
        S_pool=s_pool,
        tau_hat=tau_hat,
        alpha_default=0.05,
        alpha_grid=[0.01, 0.02, 0.05, 0.10, 0.20],
    )

    return gaussian, conformal, meta


def compute_baseline_novelty_scores(
    known_feats: np.ndarray,
    unknown_feats: np.ndarray,
    train_feats: dict[str, np.ndarray],
    gaussian: GaussianModel,
) -> dict[str, dict[str, float]]:
    """Compute AUROC, AUPR, FPR95 for all baseline models."""
    class_names = list(train_feats.keys())
    # 1. NovaSight (inflated Mahalanobis)
    def nov_score(X):
        return np.array([min(gaussian.inflated_score(x, c_idx) for c_idx in range(len(class_names))) for x in X])

    # 2. Raw Mahalanobis without inflation
    def mahal_score(X):
        return np.array([min(float((x - gaussian.class_means[c_idx][0]) @ gaussian.A @ (x - gaussian.class_means[c_idx][0])) for c_idx in range(len(class_names))) for x in X])

    # 3. Cosine centroid distance (1 - max cosine similarity to class centroids)
    centroids = np.array([train_feats[c].mean(axis=0) for c in class_names])
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    norm_centroids = centroids / c_norms

    def cosine_score(X):
        x_norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        sims = (X / x_norms) @ norm_centroids.T
        return 1.0 - np.max(sims, axis=1)

    # 4. k-NN (k=5 distance to nearest neighbor in train)
    X_train_all = np.vstack([train_feats[c] for c in class_names])
    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(X_train_all)

    def knn_score(X):
        dists, _ = knn.kneighbors(X)
        return dists[:, -1]  # 5th neighbor distance

    # 5. MSP (Maximum Softmax Probability inverted: 1 - MSP)
    def msp_score(X):
        # Softmax on negative Mahalanobis distances
        dists = np.array([[gaussian.inflated_score(x, c_idx) for c_idx in range(len(class_names))] for x in X])
        exp_neg = np.exp(-0.5 * dists)
        probs = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)
        return 1.0 - np.max(probs, axis=1)

    baselines = {
        "NovaSight": nov_score,
        "Mahalanobis_raw": mahal_score,
        "Cosine_centroid": cosine_score,
        "kNN_5": knn_score,
        "MSP": msp_score,
    }

    res = {}
    for name, fn in baselines.items():
        k_s = fn(known_feats)
        u_s = fn(unknown_feats)
        pt_auc, lo_auc, hi_auc = compute_bootstrap_ci(k_s, u_s, auroc_metric, n_resamples=1000)
        pt_pr, lo_pr, hi_pr = compute_bootstrap_ci(k_s, u_s, aupr_metric, n_resamples=1000)
        pt_fpr, lo_fpr, hi_fpr = compute_bootstrap_ci(k_s, u_s, fpr95_metric, n_resamples=1000)
        res[name] = {
            "AUROC": pt_auc, "AUROC_CI": [lo_auc, hi_auc],
            "AUPR": pt_pr, "AUPR_CI": [lo_pr, hi_pr],
            "FPR95": pt_fpr, "FPR95_CI": [lo_fpr, hi_fpr],
        }
    return res


def evaluate_run(
    held_out: str,
    split_seed: int,
    protocol: str = "v1.1",
    manifest_version: str = "v2",
    device_arg: str = "auto",
    artifact_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute evaluation for one run and return complete results dict."""
    device = setup_device(device_arg)
    run_id = f"H={held_out}_s={split_seed}"

    if artifact_dir is None:
        artifact_dir = REPO_ROOT / "artifacts" / "runs" / protocol / run_id
    if out_dir is None:
        out_dir = REPO_ROOT / "reports" / protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    gaussian, conformal, meta = load_artifact(artifact_dir)
    manifest_path = REPO_ROOT / "data" / "manifests" / manifest_version / f"{run_id}.json"
    manifest = load_manifest_v2(manifest_path)

    held_out_prefix = CLASS_TO_PREFIX.get(held_out, held_out)
    known_prefixes = [p for p in KNOWN_PREFIXES if p != held_out_prefix]
    known_classes = [PREFIX_TO_CLASS[p] for p in known_prefixes]

    backbone_id = meta["chosen_config"]["backbone"]
    feature_mode = meta["chosen_config"]["feature"]
    extractor = make_extractor(backbone_id)
    extractor.to(device)

    # 1. Load train features for baselines (read allowed train)
    with PoolGuard(allowed_pools=["train"]):
        train_feats = {}
        for p in known_prefixes:
            c_name = PREFIX_TO_CLASS[p]
            recs = get_pool_v2(manifest, p, "train")
            train_feats[c_name] = get_or_extract_features(
                records=recs, extractor=extractor, backbone_id=backbone_id,
                feature=feature_mode, manifest_version=manifest_version,
                run_id=run_id, pool="train", class_name=c_name, device=device,
            )

    # 2. Evaluate on test pools under PoolGuard
    allowed_eval_pools = ["test", "unk_test", "enroll_pool", "new_test"]
    with PoolGuard(allowed_pools=allowed_eval_pools):
        # Known test
        known_test_feats = []
        known_test_labels = []
        known_test_shas = []
        for p in known_prefixes:
            c_name = PREFIX_TO_CLASS[p]
            recs = get_pool_v2(manifest, p, "test")
            feats = get_or_extract_features(
                records=recs, extractor=extractor, backbone_id=backbone_id,
                feature=feature_mode, manifest_version=manifest_version,
                run_id=run_id, pool="test", class_name=c_name, device=device,
            )
            known_test_feats.append(feats)
            known_test_labels.extend([c_name] * len(feats))
            known_test_shas.extend([r.get("sha256", "") for r in recs])
        known_test_feats = np.concatenate(known_test_feats, axis=0)

        # Unknown test
        unk_recs = get_pool_v2(manifest, held_out_prefix, "unk_test")
        unk_test_feats = get_or_extract_features(
            records=unk_recs, extractor=extractor, backbone_id=backbone_id,
            feature=feature_mode, manifest_version=manifest_version,
            run_id=run_id, pool="unk_test", class_name=PREFIX_TO_CLASS[held_out_prefix], device=device,
        )
        unk_test_shas = [r.get("sha256", "") for r in unk_recs]

        # Table T1: Closed-set accuracy on known test
        closed_preds = []
        pred_sets_005 = []
        for f_vec in known_test_feats:
            dt_dict = {c: gaussian.inflated_score(f_vec, i) for i, c in enumerate(gaussian.class_names)}
            c_hat = min(dt_dict, key=dt_dict.get)
            closed_preds.append(c_hat)
            p_set, _ = conformal.prediction_set(dt_dict, alpha=0.05)
            pred_sets_005.append(p_set)

        acc = float(accuracy_score(known_test_labels, closed_preds))
        prec, rec, f1, _ = precision_recall_fscore_support(known_test_labels, closed_preds, labels=known_classes, zero_division=0)
        table_t1 = {
            "closed_set_accuracy": acc,
            "per_class": {
                c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
                for i, c in enumerate(known_classes)
            },
        }

        # Table T2: Baselines & novelty metrics
        table_t2 = compute_baseline_novelty_scores(known_test_feats, unk_test_feats, train_feats, gaussian)

        # Table T3: Decision rates at alpha=0.05
        # Also score unk_test with prediction sets
        unk_pred_sets_005 = []
        unk_closed_preds = []
        for f_vec in unk_test_feats:
            dt_dict = {c: gaussian.inflated_score(f_vec, i) for i, c in enumerate(gaussian.class_names)}
            c_hat = min(dt_dict, key=dt_dict.get)
            unk_closed_preds.append(c_hat)
            p_set, _ = conformal.prediction_set(dt_dict, alpha=0.05)
            unk_pred_sets_005.append(p_set)

        all_pred_sets = pred_sets_005 + unk_pred_sets_005
        all_true_labels = known_test_labels + [UNKNOWN_LABEL] * len(unk_test_feats)
        all_is_known = [True] * len(known_test_labels) + [False] * len(unk_test_feats)
        all_closed_preds = closed_preds + unk_closed_preds

        decision_rates = compute_decision_rates(all_pred_sets, all_true_labels, all_is_known)
        cm = confusion_matrix_v11(all_true_labels, all_pred_sets, all_closed_preds, known_classes)
        table_t3 = {
            "decision_rates": decision_rates,
            "confusion_matrix": cm["matrix"],
            "markdown_table": cm["markdown_table"],
        }

        # Table T4: Coverage across alpha grid + Beta spread
        table_t4 = {"grid": {}, "beta_spread": compute_beta_spread(n=40, alpha=0.05)}
        for alpha in conformal.alpha_grid:
            cov_known = 0
            for f_vec, true_c in zip(known_test_feats, known_test_labels):
                dt_dict = {c: gaussian.inflated_score(f_vec, i) for i, c in enumerate(gaussian.class_names)}
                p_set, _ = conformal.prediction_set(dt_dict, alpha=alpha)
                if true_c in p_set:
                    cov_known += 1
            table_t4["grid"][str(alpha)] = float(cov_known / len(known_test_labels))

        # Table T5: 10 enrollment draws from enroll_pool, evaluated on new_test
        enroll_recs = get_pool_v2(manifest, held_out_prefix, "enroll_pool")
        new_test_recs = get_pool_v2(manifest, held_out_prefix, "new_test")

        enroll_feats = get_or_extract_features(
            records=enroll_recs, extractor=extractor, backbone_id=backbone_id,
            feature=feature_mode, manifest_version=manifest_version,
            run_id=run_id, pool="enroll_pool", class_name=PREFIX_TO_CLASS[held_out_prefix], device=device,
        )
        new_test_feats = get_or_extract_features(
            records=new_test_recs, extractor=extractor, backbone_id=backbone_id,
            feature=feature_mode, manifest_version=manifest_version,
            run_id=run_id, pool="new_test", class_name=PREFIX_TO_CLASS[held_out_prefix], device=device,
        )

        gate_params = meta.get("gate_params", {"alpha_cons": 0.01, "q_BH": 0.05, "eps_max": 0.10})
        draw_results = []
        rng = np.random.default_rng(split_seed)
        n_enroll = len(enroll_feats)

        for draw_idx in range(10):
            chosen_indices = rng.choice(n_enroll, size=5, replace=False)
            exemplar_feats = enroll_feats[chosen_indices]
            exemplar_shas = [enroll_recs[idx].get("sha256", f"sha_{idx}") for idx in chosen_indices]

            # Run enrollment gate (recorded, never blocks evaluation per spec)
            gate_rep = run_gate(
                embeddings=exemplar_feats,
                sha256s=exemplar_shas,
                class_name=PREFIX_TO_CLASS[held_out_prefix],
                existing_class_names=gaussian.class_names,
                gaussian=gaussian,
                conformal=conformal,
                alpha_cons=gate_params["alpha_cons"],
                q_BH=gate_params["q_BH"],
                eps_max=gate_params["eps_max"],
            )

            # Evaluate H1 and H1-s on new_test
            mu_new = exemplar_feats.mean(axis=0)
            diffs = [exemplar_feats[i] - (exemplar_feats.sum(axis=0) - exemplar_feats[i]) / 4.0 for i in range(5)]
            loo_dists = [float(d @ gaussian.A @ d) / 1.25 for d in diffs]
            r_hat = max(1e-6, (float(np.mean(loo_dists))) / max(conformal.tau_hat, 1e-6))

            # New test evaluation
            eval_diffs = new_test_feats - mu_new
            raw_dists = np.sum((eval_diffs @ gaussian.A) * eval_diffs, axis=1)

            scores_h1 = raw_dists / 1.2
            scores_h1s = (raw_dists / 1.2) / r_hat

            p_vals_h1 = np.array([conformal.p_value_from_pool(s) for s in scores_h1])
            p_vals_h1s = np.array([conformal.p_value_from_pool(s) for s in scores_h1s])

            cov_h1 = float(np.mean(p_vals_h1 > 0.05))
            cov_h1s = float(np.mean(p_vals_h1s > 0.05))

            # McNemar on known test: evaluate if known accuracy changes after enrolling new class
            # When new class is enrolled, c_hat can be new class if dt_new < min_known
            preds_after = []
            for f_vec in known_test_feats:
                dt_known = {c: gaussian.inflated_score(f_vec, i) for i, c in enumerate(gaussian.class_names)}
                min_k = min(dt_known.values())
                diff_n = f_vec - mu_new
                dt_new = float(diff_n @ gaussian.A @ diff_n) / 1.2
                if dt_new < min_k:
                    preds_after.append(PREFIX_TO_CLASS[held_out_prefix])
                else:
                    preds_after.append(min(dt_known, key=dt_known.get))

            mcnemar = paired_mcnemar_test(closed_preds, preds_after, known_test_labels)

            draw_results.append({
                "draw_idx": draw_idx,
                "draw_indices": chosen_indices.tolist(),
                "exemplar_shas": exemplar_shas,
                "gate_outcome": gate_rep.outcome,
                "gate_notes": gate_rep.notes,
                "H1_coverage": cov_h1,
                "H1_FRR": 1.0 - cov_h1,
                "H1s_coverage": cov_h1s,
                "H1s_FRR": 1.0 - cov_h1s,
                "mcnemar": mcnemar,
            })

        table_t5 = {
            "draws": draw_results,
            "mean_H1_coverage": float(np.mean([d["H1_coverage"] for d in draw_results])),
            "mean_H1_FRR": float(np.mean([d["H1_FRR"] for d in draw_results])),
            "mean_H1s_coverage": float(np.mean([d["H1s_coverage"] for d in draw_results])),
            "mean_H1s_FRR": float(np.mean([d["H1s_FRR"] for d in draw_results])),
        }

    out_record = {
        "run_id": run_id,
        "held_out": held_out,
        "split_seed": split_seed,
        "protocol": protocol,
        "manifest_version": manifest_version,
        "artifact_dir": str(artifact_dir),
        "timestamp": time.time(),
        "T1": table_t1,
        "T2": table_t2,
        "T3": table_t3,
        "T4": table_t4,
        "T5": table_t5,
    }

    out_file = out_dir / f"results_{run_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_record, f, indent=2)

    cm_file = out_dir / f"confusion_matrix_{run_id}.json"
    with open(cm_file, "w", encoding="utf-8") as f:
        json.dump(cm, f, indent=2)

    logger.info("Evaluation complete for %s. Results written to %s", run_id, out_file)
    return out_record


def main() -> None:
    setup_logging("INFO")
    parser = argparse.ArgumentParser(description="Evaluate NovaSight artifact under Protocol v1.1")
    parser.add_argument("--held-out", default="Sc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", default="v1.1")
    parser.add_argument("--manifest-version", default="v2")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    evaluate_run(
        held_out=args.held_out,
        split_seed=args.seed,
        protocol=args.protocol,
        manifest_version=args.manifest_version,
        device_arg=args.device,
    )


if __name__ == "__main__":
    main()
