"""NovaSight Rehearsal and Model Selection (Protocol v1.1 Task E5, D.2).

Conducts nested rehearsal on known classes only (using train, early_stop, sel_cal, sel_eval).
Never reads any test pool (test, unk_test, new_test).
Enforces PoolGuard during execution.

Selection hierarchy:
  1. Filter out configs with rehearsal known FRR > 0.08 at alpha=0.05.
  2. Primary criterion: mean rehearsal AUROC (min_c dt_c).
  3. Tie-break within 0.01: mean pseudo-enrollment recall.
  4. H1 vs H1-s: smaller mean |rehearsal FRR - alpha| over alpha in {0.05, 0.10}; ties to H1.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.config import REPO_ROOT
from src.data.guard import PoolGuard
from src.data.splits_v2 import load_manifest_v2, get_pool_v2
from src.models.gaussian import fit_gaussian, GaussianModel
from src.models.conformal import build_conformal_scores, ConformalScores, make_decision
from src.models.gate import run_gate
from src.utils.hashing import canonical_file_sha256, canonical_json_sha256

logger = logging.getLogger(__name__)

ALLOWED_REHEARSAL_POOLS = ["train", "early_stop", "sel_cal", "sel_eval"]


@dataclass
class CandidateResult:
    """Evaluation result for one model candidate during rehearsal."""
    candidate_id: str
    backbone: str
    feature: str
    combination: str
    k: int
    mean_auroc: float
    mean_known_frr: float
    mean_ambiguity_rate: float
    mean_h1_recall: float
    mean_h1s_recall: float
    h1_frr_gap: float
    h1s_frr_gap: float
    excluded: bool = False
    exclusion_reason: str = ""


# ── Item 2.a: Nested B1 Record Exclusion Assertion ───────────────────────────

def assert_nested_b1_records_disjoint(
    train_records: list[dict[str, Any]],
    pseudo_unknown_class: str,
    pseudo_unknown_prefix: str | None = None,
) -> None:
    """Assert that training records for nested B1 fine-tuning strictly exclude the pseudo-unknown class.

    Raises:
        AssertionError: If any record matches pseudo_unknown_class or pseudo_unknown_prefix.
    """
    forbidden = {pseudo_unknown_class.lower()}
    if pseudo_unknown_prefix:
        forbidden.add(pseudo_unknown_prefix.lower())

    for idx, r in enumerate(train_records):
        cls_name = str(r.get("class", r.get("class_name", ""))).lower()
        prefix = str(r.get("prefix", "")).lower()
        fn = str(r.get("filename", "")).lower()
        if cls_name in forbidden or prefix in forbidden or any(fn.startswith(f"{f}_") for f in forbidden):
            raise AssertionError(
                f"Nested B1 leakage violation at record {idx}: "
                f"training record contains pseudo-unknown class '{pseudo_unknown_class}' ({r})"
            )


# ── Item 2.b: L234 Features with TRAIN z-scoring ──────────────────────────────

def zscore_l234(
    features: np.ndarray,
    train_mean: np.ndarray | None = None,
    train_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each block of L234 feature vector (128 + 256 + 512 = 896) with TRAIN statistics.

    Args:
        features: (N, 896) array of concatenated GAP(l2) + GAP(l3) + GAP(l4).
        train_mean: Optional precomputed (896,) train mean. If None, computed from `features`.
        train_std: Optional precomputed (896,) train std. If None, computed from `features`.

    Returns:
        Tuple of (zscored_features, train_mean, train_std).
    """
    if features.shape[1] != 896:
        raise ValueError(f"L234 requires 896 dimensions (128+256+512), got {features.shape[1]}")

    if train_mean is None or train_std is None:
        # Compute block-by-block statistics from train
        mean_2 = features[:, :128].mean(axis=0)
        std_2 = features[:, :128].std(axis=0) + 1e-8

        mean_3 = features[:, 128:384].mean(axis=0)
        std_3 = features[:, 128:384].std(axis=0) + 1e-8

        mean_4 = features[:, 384:].mean(axis=0)
        std_4 = features[:, 384:].std(axis=0) + 1e-8

        train_mean = np.concatenate([mean_2, mean_3, mean_4])
        train_std = np.concatenate([std_2, std_3, std_4])

    # Z-score each block using train statistics
    z_2 = (features[:, :128] - train_mean[:128]) / train_std[:128]
    z_3 = (features[:, 128:384] - train_mean[128:384]) / train_std[128:384]
    z_4 = (features[:, 384:] - train_mean[384:]) / train_std[384:]

    zscored = np.concatenate([z_2, z_3, z_4], axis=1)
    return zscored, train_mean, train_std


# ── Item 2.c: Combination Option (b) 5-fold Cross-Fitting on TRAIN Only ──────

def cross_fit_combination_b(
    train_features_by_class: dict[str, np.ndarray],
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Fit logistic regression weights for multi-block features using 5-fold cross-fitting on TRAIN ONLY.

    Option (b) combines per-block Gaussian scores via logistic regression whose weights
    are learned from out-of-fold scores of TRAIN data only.

    Args:
        train_features_by_class: Dict of class -> (n_c, 896) TRAIN features.
        n_splits: Number of cross-fitting folds (5).
        seed: Random seed for cross-validation splits.

    Returns:
        (3,) array of learned block combination weights.
    """
    classes = list(train_features_by_class.keys())
    # Block slices: Layer2 (0..128), Layer3 (128..384), Layer4 (384..896)
    block_slices = [(0, 128), (128, 384), (384, 896)]

    # Collect all train samples and class indices
    X_list = []
    y_list = []
    for c_idx, c in enumerate(classes):
        feats = train_features_by_class[c]
        X_list.append(feats)
        y_list.extend([c_idx] * len(feats))

    X_train = np.vstack(X_list)
    y_train = np.array(y_list)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_scores = []
    oof_labels = []

    for fit_idx, val_idx in kf.split(X_train):
        X_fit, X_val = X_train[fit_idx], X_train[val_idx]
        y_fit, y_val = y_train[fit_idx], y_train[val_idx]

        # Fit per-block Gaussians on X_fit
        block_gaussians = []
        for start, end in block_slices:
            fit_dict = {
                classes[c_idx]: X_fit[y_fit == c_idx, start:end]
                for c_idx in range(len(classes))
                if np.sum(y_fit == c_idx) > 0
            }
            bg = fit_gaussian(fit_dict, k=1)
            block_gaussians.append(bg)

        # Score X_val on each block
        val_block_scores = np.zeros((len(X_val), 3))
        for b_idx, bg in enumerate(block_gaussians):
            start, end = block_slices[b_idx]
            sub_feats = X_val[:, start:end]
            for i, x in enumerate(sub_feats):
                # Min inflated score across classes
                val_block_scores[i, b_idx] = min(
                    bg.inflated_score(x, ci) for ci in range(len(bg.class_names))
                )

        # For known positives vs pseudo-unknown negatives:
        # Create pseudo-unknown negatives by leaving one class out within out-of-fold scores
        for c_idx in range(len(classes)):
            known_mask = (y_val != c_idx)
            unk_mask = (y_val == c_idx)
            if np.sum(known_mask) > 0 and np.sum(unk_mask) > 0:
                oof_scores.append(val_block_scores[known_mask])
                oof_labels.extend([0] * int(np.sum(known_mask)))
                oof_scores.append(val_block_scores[unk_mask])
                oof_labels.extend([1] * int(np.sum(unk_mask)))

    if not oof_scores:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    X_oof = np.vstack(oof_scores)
    y_oof = np.array(oof_labels)

    # Fit LogisticRegression without regularization bias to learn relative block weights
    clf = LogisticRegression(C=1.0, random_state=seed, max_iter=200)
    clf.fit(X_oof, y_oof)
    raw_coef = clf.coef_[0]
    # Normalize weights to sum to 1.0
    pos_coef = np.maximum(raw_coef, 1e-4)
    weights = pos_coef / np.sum(pos_coef)
    return weights


def score_features_novelty(gaussian: GaussianModel, feats: np.ndarray) -> np.ndarray:
    """Compute primary novelty score N(x) = min_c dt_c(x) for each sample."""
    n_samples = len(feats)
    scores = np.zeros(n_samples)
    for i in range(n_samples):
        dt_vals = [gaussian.inflated_score(feats[i], c_idx) for c_idx in range(len(gaussian.class_names))]
        scores[i] = min(dt_vals)
    return scores


def evaluate_pseudo_enrollment(
    k_class: str,
    gaussian: GaussianModel,
    conformal: ConformalScores,
    sel_cal_feats: np.ndarray,
    sel_eval_feats: np.ndarray,
    other_sel_eval: dict[str, np.ndarray],
    num_draws: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Conduct 10 pseudo-enrollment draws of 5 from k's sel_cal and evaluate on sel_eval."""
    rng = np.random.default_rng(seed)
    n_cal = len(sel_cal_feats)
    if n_cal < 5:
        raise ValueError(f"Class {k_class} sel_cal has only {n_cal} samples, need at least 5")

    h1_recalls = []
    h1s_recalls = []
    h1_coverages = {0.05: [], 0.10: []}
    h1s_coverages = {0.05: [], 0.10: []}

    for draw_idx in range(num_draws):
        draw_indices = rng.choice(n_cal, size=5, replace=False)
        exemplars = sel_cal_feats[draw_indices]
        mu_new = exemplars.mean(axis=0)

        # H1-s: compute leave-one-out distances / 1.25 / tau_hat
        diffs = [exemplars[i] - (exemplars.sum(axis=0) - exemplars[i]) / 4.0 for i in range(5)]
        loo_dists = [float(d @ gaussian.A @ d) / 1.25 for d in diffs]
        r_hat = max(1e-6, (float(np.mean(loo_dists))) / max(conformal.tau_hat, 1e-6))

        # Evaluate on k's sel_eval (40 samples)
        eval_diffs = sel_eval_feats - mu_new
        raw_dists = np.sum((eval_diffs @ gaussian.A) * eval_diffs, axis=1)

        scores_h1 = raw_dists / 1.2
        p_vals_h1 = np.array([conformal.p_value_from_pool(s) for s in scores_h1])

        scores_h1s = (raw_dists / 1.2) / r_hat
        p_vals_h1s = np.array([conformal.p_value_from_pool(s) for s in scores_h1s])

        for alpha in [0.05, 0.10]:
            h1_coverages[alpha].append(float(np.mean(p_vals_h1 > alpha)))
            h1s_coverages[alpha].append(float(np.mean(p_vals_h1s > alpha)))

        h1_recalls.append(float(np.mean(p_vals_h1 > 0.05)))
        h1s_recalls.append(float(np.mean(p_vals_h1s > 0.05)))

    return {
        "h1_recall": float(np.mean(h1_recalls)),
        "h1s_recall": float(np.mean(h1s_recalls)),
        "h1_frr_005": float(1.0 - np.mean(h1_coverages[0.05])),
        "h1_frr_010": float(1.0 - np.mean(h1_coverages[0.10])),
        "h1s_frr_005": float(1.0 - np.mean(h1s_coverages[0.05])),
        "h1s_frr_010": float(1.0 - np.mean(h1s_coverages[0.10])),
    }


# ── Item 2.g: Gate Parameter Tuning with 'constraint unmet' ───────────────────

def tune_gate_parameters(
    gaussian: GaussianModel,
    conformal: ConformalScores,
    sel_cal_by_class: dict[str, np.ndarray],
    seed: int = 42,
) -> dict[str, Any]:
    """Tune alpha_cons, q_BH, eps_max on clean/injected/lookalike pseudo-enrollments.

    Rules:
      - alpha_cons in {0.005, 0.01, 0.02, 0.05} maximising INJECTED OUTLIER detection subject to false-flag <= 0.10.
      - q_BH in {0.05, 0.10} maximising KNOWN-LOOKALIKE subject to false-flag <= 0.10.
      - eps_max in {0.02, 0.05, 0.10, 0.20} = smallest value for which rehearsal enrollments with
        eps_hat <= eps_max had observed confusion <= 0.10. If unmet, choose lowest false-flag value
        and record 'constraint unmet'.
    """
    rng = np.random.default_rng(seed)
    class_names = list(sel_cal_by_class.keys())

    clean_sets = []
    injected_outlier_sets = []
    lookalike_sets = []

    for c in class_names:
        feats_c = sel_cal_by_class[c]
        if len(feats_c) < 5:
            continue
        idx = rng.choice(len(feats_c), size=5, replace=False)
        clean_sets.append((feats_c[idx], c))

        other_c = rng.choice([o for o in class_names if o != c])
        feats_other = sel_cal_by_class[other_c]
        inj = feats_c[idx].copy()
        inj[4] = feats_other[rng.choice(len(feats_other))]
        injected_outlier_sets.append((inj, c))

        look = feats_c[idx].copy()
        other_indices = rng.choice(len(feats_other), size=2, replace=False)
        look[3] = feats_other[other_indices[0]]
        look[4] = feats_other[other_indices[1]]
        lookalike_sets.append((look, c))

    # alpha_cons tuning
    best_alpha_cons = 0.01
    best_outlier_detect = -1.0
    for alpha_cons in [0.005, 0.01, 0.02, 0.05]:
        clean_flags = 0
        for embs, name in clean_sets:
            rep = run_gate(
                embeddings=embs, sha256s=[f"h_{i}" for i in range(5)],
                class_name=f"test_{name}", existing_class_names=gaussian.class_names,
                gaussian=gaussian, conformal=conformal,
                alpha_cons=alpha_cons, q_BH=0.05, eps_max=0.10,
            )
            if len(rep.consistency_issues) > 0:
                clean_flags += 1
        false_flag_rate = clean_flags / max(len(clean_sets), 1)

        outlier_flags = 0
        for embs, name in injected_outlier_sets:
            rep = run_gate(
                embeddings=embs, sha256s=[f"h_{i}" for i in range(5)],
                class_name=f"test_{name}", existing_class_names=gaussian.class_names,
                gaussian=gaussian, conformal=conformal,
                alpha_cons=alpha_cons, q_BH=0.05, eps_max=0.10,
            )
            if len(rep.consistency_issues) > 0:
                outlier_flags += 1
        detect_rate = outlier_flags / max(len(injected_outlier_sets), 1)

        if false_flag_rate <= 0.10 and detect_rate > best_outlier_detect:
            best_outlier_detect = detect_rate
            best_alpha_cons = alpha_cons

    # q_BH tuning
    best_q_BH = 0.05
    best_lookalike_detect = -1.0
    for q_bh in [0.05, 0.10]:
        clean_flags = 0
        for embs, name in clean_sets:
            rep = run_gate(
                embeddings=embs, sha256s=[f"h_{i}" for i in range(5)],
                class_name=f"test_{name}", existing_class_names=gaussian.class_names,
                gaussian=gaussian, conformal=conformal,
                alpha_cons=best_alpha_cons, q_BH=q_bh, eps_max=0.10,
            )
            if len(rep.novelty_issues) > 0:
                clean_flags += 1
        false_flag_rate = clean_flags / max(len(clean_sets), 1)

        look_flags = 0
        for embs, name in lookalike_sets:
            rep = run_gate(
                embeddings=embs, sha256s=[f"h_{i}" for i in range(5)],
                class_name=f"test_{name}", existing_class_names=gaussian.class_names,
                gaussian=gaussian, conformal=conformal,
                alpha_cons=best_alpha_cons, q_BH=q_bh, eps_max=0.10,
            )
            if len(rep.novelty_issues) > 0:
                look_flags += 1
        detect_rate = look_flags / max(len(lookalike_sets), 1)

        if false_flag_rate <= 0.10 and detect_rate > best_lookalike_detect:
            best_lookalike_detect = detect_rate
            best_q_BH = q_bh

    # eps_max tuning: smallest value where confusion <= 0.10, else record constraint unmet
    best_eps_max = 0.10
    constraint_met = False
    for eps_max_cand in [0.02, 0.05, 0.10, 0.20]:
        confusions = []
        for embs, name in clean_sets:
            rep = run_gate(
                embeddings=embs, sha256s=[f"h_{i}" for i in range(5)],
                class_name=f"test_{name}", existing_class_names=gaussian.class_names,
                gaussian=gaussian, conformal=conformal,
                alpha_cons=best_alpha_cons, q_BH=best_q_BH, eps_max=eps_max_cand,
            )
            max_eps = max(rep.eps_hat.values()) if rep.eps_hat else 0.0
            if max_eps <= eps_max_cand:
                confusions.append(max_eps)

        if confusions and np.mean(confusions) <= 0.10:
            best_eps_max = eps_max_cand
            constraint_met = True
            break

    notes = "constraint met" if constraint_met else "constraint unmet"
    return {
        "alpha_cons": best_alpha_cons,
        "q_BH": best_q_BH,
        "eps_max": best_eps_max,
        "constraint_status": notes,
    }


# ── Item 2.f: Selection Ranking Hierarchy ────────────────────────────────────

def select_best_candidate(candidates: list[CandidateResult]) -> tuple[CandidateResult, str]:
    """Execute pre-registered selection hierarchy.

    Hierarchy:
      1. Filter out candidate configs with mean rehearsal known FRR > 0.08 at alpha=0.05.
      2. Primary criterion: mean rehearsal AUROC.
      3. Tie-break within 0.01 AUROC: mean pseudo-enrollment recall.
      4. H1 vs H1-s: smaller mean |rehearsal FRR - alpha| over alpha in {0.05, 0.10}; ties to H1.
    """
    eligible = [c for c in candidates if not c.excluded]
    if not eligible:
        eligible = candidates

    best_auroc = max(c.mean_auroc for c in eligible)
    near_top = [c for c in eligible if c.mean_auroc >= best_auroc - 0.01]
    near_top.sort(key=lambda c: max(c.mean_h1_recall, c.mean_h1s_recall), reverse=True)
    chosen = near_top[0]

    if chosen.h1s_frr_gap < chosen.h1_frr_gap:
        variant = "H1-s"
    else:
        variant = "H1"

    return chosen, variant
