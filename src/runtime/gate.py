"""NovaSight five-shot enrollment gate (torch-free runtime implementation).

Performs the four enrollment gate checks:
  V (validity): name non-empty & unique, no duplicate sha256, no near-dup embeddings
  N (novelty): Benjamini-Hochberg test against known classes
  C (consistency): leave-one-out distance check
  O (overlap): Delta2-based confusion estimate

Outcomes:
  ACCEPT - all checks pass
  REVIEW - warnings raised (requires operator confirmation)
  REJECT - critical failure (blocks enrollment)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.runtime.scoring import RuntimeScorer


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function using built-in math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

logger = logging.getLogger(__name__)


@dataclass
class GateReport:
    """Result of running the enrollment gate."""

    outcome: str   # "ACCEPT" | "REVIEW" | "REJECT"
    validity_issues: list[str] = field(default_factory=list)
    novelty_issues: list[str] = field(default_factory=list)
    consistency_issues: list[str] = field(default_factory=list)
    overlap_issues: list[str] = field(default_factory=list)
    eps_hat: dict[str, float] = field(default_factory=dict)
    eps_hat_bootstrap: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON storage."""
        return {
            "outcome": self.outcome,
            "validity_issues": self.validity_issues,
            "novelty_issues": self.novelty_issues,
            "consistency_issues": self.consistency_issues,
            "overlap_issues": self.overlap_issues,
            "eps_hat": self.eps_hat,
            "eps_hat_bootstrap": {
                k: list(v) for k, v in self.eps_hat_bootstrap.items()
            },
            "notes": self.notes,
        }


def _bh_correction(
    p_values: np.ndarray,
    q: float,
) -> np.ndarray:
    """Apply Benjamini-Hochberg multiple testing correction.

    Args:
        p_values: Array of p-values.
        q: BH false discovery rate level.

    Returns:
        Boolean array: True if the hypothesis is rejected.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = q * np.arange(1, n + 1) / n
    reject_up_to = -1
    for k in range(n - 1, -1, -1):
        if sorted_p[k] <= thresholds[k]:
            reject_up_to = k
            break
    result = np.zeros(n, dtype=bool)
    if reject_up_to >= 0:
        result[order[:reject_up_to + 1]] = True
    return result


def run_gate(
    embeddings: np.ndarray,              # (5, D) — 5 example embeddings
    sha256s: list[str],                  # sha256 of each of the 5 images
    class_name: str,                     # proposed class name
    existing_class_names: list[str],     # names already in memory (known + enrolled)
    scorer: RuntimeScorer,               # pure-numpy runtime scoring engine
    alpha_cons: float,
    q_BH: float,
    eps_max: float,
    existing_embeddings: list[np.ndarray] | None = None,  # known class embeddings
    near_dup_cos_threshold: float = 0.95,
) -> GateReport:
    """Run all four enrollment gate checks using RuntimeScorer.

    Args:
        embeddings: (5, D) feature matrix for the 5 examples.
        sha256s: SHA-256 hashes of the 5 images.
        class_name: Proposed name for the new class.
        existing_class_names: Names of classes already enrolled.
        scorer: RuntimeScorer instance.
        alpha_cons: Consistency check threshold.
        q_BH: Benjamini-Hochberg FDR level.
        eps_max: Max confusion estimate before REVIEW.
        existing_embeddings: Optional list of all known/enrolled embeddings.
        near_dup_cos_threshold: Cosine similarity threshold for near-duplicate detection.

    Returns:
        GateReport with outcome and per-check issues.
    """
    report = GateReport(outcome="ACCEPT")

    # ── V: Validity ───────────────────────────────────────────────────────────
    if not class_name or not class_name.strip():
        report.validity_issues.append("Name is empty.")
        report.outcome = "REJECT"

    if class_name in existing_class_names:
        report.validity_issues.append(f"Name '{class_name}' already exists.")
        report.outcome = "REJECT"

    if len(set(sha256s)) < len(sha256s):
        report.validity_issues.append("Duplicate images detected (same sha256).")
        report.outcome = "REJECT"

    for i in range(5):
        for j in range(i + 1, 5):
            cos = float(
                np.dot(embeddings[i], embeddings[j])
                / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-12)
            )
            if cos >= near_dup_cos_threshold:
                report.validity_issues.append(
                    f"Examples {i+1} and {j+1} are near-duplicates (cosine={cos:.3f})."
                )
                report.outcome = "REJECT"

    if existing_embeddings and len(existing_embeddings) > 0:
        all_existing = np.vstack(existing_embeddings)
        for i, emb in enumerate(embeddings):
            norms = (np.linalg.norm(all_existing, axis=1) * np.linalg.norm(emb)) + 1e-12
            cos_sims = (all_existing @ emb) / norms
            if cos_sims.max() >= near_dup_cos_threshold:
                report.validity_issues.append(
                    f"Example {i+1} is a near-duplicate of an existing stored image."
                )
                report.outcome = "REJECT"

    if report.outcome == "REJECT":
        report.notes = "REJECTED: validity failure. " + "; ".join(report.validity_issues)
        return report

    # ── N: Novelty (BH test) ──────────────────────────────────────────────────
    K = len(scorer.class_names)
    all_p_values = np.zeros((5, K), dtype=np.float64)
    for i in range(5):
        for c_idx, c in enumerate(scorer.class_names):
            score = scorer.inflated_score(embeddings[i], c_idx)
            all_p_values[i, c_idx] = scorer.conformal_p_value(score, c)

    flat_p = all_p_values.flatten()
    rejected = _bh_correction(flat_p, q_BH).reshape(5, K)

    consistent_with: dict[str, int] = {}
    for c_idx, c in enumerate(scorer.class_names):
        consistent_count = int((~rejected[:, c_idx]).sum())
        consistent_with[c] = consistent_count

    for c, count in consistent_with.items():
        if count >= 3:
            report.novelty_issues.append(
                f"REJECT: {count}/5 examples are consistent with known class '{c}'. "
                f"Consider labelling them as '{c}'."
            )
            report.outcome = "REJECT"
        elif 1 <= count <= 2:
            c_idx = scorer.class_names.index(c)
            flagged_indices = [i + 1 for i in range(5) if not rejected[i, c_idx]]
            report.novelty_issues.append(
                f"REVIEW: {count}/5 examples may be consistent with class '{c}' "
                f"(examples: {flagged_indices})."
            )
            if report.outcome == "ACCEPT":
                report.outcome = "REVIEW"

    if report.outcome == "REJECT":
        report.notes = "REJECTED: novelty check failure. " + "; ".join(report.novelty_issues)
        return report

    # ── C: Consistency ────────────────────────────────────────────────────────
    mu_new = embeddings.mean(axis=0)
    for i in range(5):
        mu_others = (embeddings.sum(axis=0) - embeddings[i]) / 4.0
        diff = embeddings[i] - mu_others
        d = float(diff @ scorer.A @ diff)
        dt = d / (1.0 + 1.0 / 4)
        p_c = scorer.p_value_from_pool(dt)
        if p_c < alpha_cons:
            report.consistency_issues.append(
                f"REVIEW: Example {i+1} is unlike the other four "
                f"(p={p_c:.4f} < alpha_cons={alpha_cons})."
            )
            if report.outcome == "ACCEPT":
                report.outcome = "REVIEW"

    # ── O: Overlap ────────────────────────────────────────────────────────────
    for c_idx, c in enumerate(scorer.class_names):
        mu_c = scorer.class_means[c_idx][0]
        n_c = scorer.n_c[c_idx]
        delta_hat = mu_new - mu_c
        raw = float(delta_hat @ scorer.A @ delta_hat)
        tau = scorer.artifact.tau_hat
        Delta2_hat = max(0.0, raw - tau * (1.0 / 5 + 1.0 / n_c))
        eps_hat = float(_normal_cdf(-np.sqrt(Delta2_hat) / 2.0))
        report.eps_hat[c] = eps_hat

        A = scorer.A
        Sigma_hat = np.linalg.inv(A)
        ASigma = A @ Sigma_hat
        var_raw = 2.0 * np.trace(ASigma @ ASigma) * (1.0 / 5 + 1.0 / n_c) ** 2
        se_raw = float(np.sqrt(max(var_raw, 0.0)))
        Delta2_lo = max(0.0, raw - se_raw - tau * (1.0 / 5 + 1.0 / n_c))
        Delta2_hi = max(0.0, raw + se_raw - tau * (1.0 / 5 + 1.0 / n_c))
        eps_lo = float(_normal_cdf(-np.sqrt(Delta2_hi) / 2.0))
        eps_hi = float(_normal_cdf(-np.sqrt(Delta2_lo) / 2.0))
        report.eps_hat_bootstrap[c] = (eps_lo, eps_hi)

        if eps_hat > eps_max:
            report.overlap_issues.append(
                f"REVIEW: Estimated confusion with class '{c}': "
                f"eps_hat={eps_hat:.3f} > eps_max={eps_max} "
                f"(bootstrap range: [{eps_lo:.3f}, {eps_hi:.3f}]). "
                f"Consider merging."
            )
            if report.outcome == "ACCEPT":
                report.outcome = "REVIEW"

    all_issues = (
        report.validity_issues + report.novelty_issues +
        report.consistency_issues + report.overlap_issues
    )
    if report.outcome == "ACCEPT":
        report.notes = "All checks passed."
    elif report.outcome == "REVIEW":
        report.notes = "Warnings: " + " | ".join(all_issues)
    else:
        report.notes = "REJECTED. " + " | ".join(all_issues)

    return report
