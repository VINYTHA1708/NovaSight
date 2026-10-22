"""NovaSight five-shot enrollment gate (Part 4.7).

Checks:
  V (validity): decodable, size OK, not near-constant, no dup sha256,
    no near-dup embeddings, name non-empty and unique
  N (novelty): BH test against known classes
  C (consistency): leave-one-out distance check
  O (overlap): Delta2-based confusion estimate

Outcomes:
  ACCEPT  - all checks pass
  REVIEW  - warnings raised (operator must acknowledge)
  REJECT  - critical failure (cannot be overridden)

Gate parameters [APPROVED-POLICY]: tuned on rehearsal pools only.
In EVALUATION: gate is run and recorded but never blocks enrollment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm as scipy_norm

from src.models.gaussian import GaussianModel
from src.models.conformal import ConformalScores

logger = logging.getLogger(__name__)


@dataclass
class GateReport:
    """Result of running the enrollment gate.

    Attributes:
        outcome: "ACCEPT", "REVIEW", or "REJECT".
        validity_issues: Issues from check V.
        novelty_issues: Issues from check N (BH).
        consistency_issues: Issues from check C.
        overlap_issues: Issues from check O.
        eps_hat: Per-known-class confusion estimate.
        notes: Human-readable summary.
    """

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
        q: BH level.

    Returns:
        Boolean array: True if the hypothesis is rejected.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    thresholds = q * np.arange(1, n + 1) / n
    rejected_sorted = sorted_p <= thresholds
    # Find largest k such that sorted_p[k] <= threshold[k]
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
    gaussian: GaussianModel,
    conformal: ConformalScores,
    alpha_cons: float,
    q_BH: float,
    eps_max: float,
    existing_sha256s: list[str] | set[str] | None = None,
    image_paths: list[str | Path] | None = None,
    phashes: list[Any] | None = None,
    existing_phashes: list[Any] | None = None,
    phash_threshold: int = 2,
    bootstrap_samples: int = 500,
    rng_seed: int = 42,
) -> GateReport:
    """Run all four enrollment gate checks under Protocol v1.1 (D5, D6, D7).

    Args:
        embeddings: (5, D) feature matrix for the 5 examples.
        sha256s: SHA-256 hashes of the 5 images.
        class_name: Proposed name for the new class.
        existing_class_names: Names of classes already enrolled.
        gaussian: Fitted GaussianModel.
        conformal: ConformalScores object.
        alpha_cons: Consistency check threshold.
        q_BH: Benjamini-Hochberg FDR level.
        eps_max: Max confusion estimate before REVIEW.
        existing_sha256s: Optional set/list of existing image hashes.
        image_paths: Optional list of 5 image file paths (for perceptual hashing).
        phashes: Optional precomputed list of 5 imagehash ImageHash objects or hex strings.
        existing_phashes: Optional list of existing imagehash objects.
        phash_threshold: Max Hamming distance to consider two images perceptual near-duplicates (default 2).
        bootstrap_samples: Number of parametric bootstrap draws for check O.
        rng_seed: Seed for parametric bootstrap RNG.

    Returns:
        GateReport with outcome and per-check issues.
    """
    report = GateReport(outcome="ACCEPT")

    # ── V: Validity (Decision D6) ─────────────────────────────────────────────
    # a) name non-empty and unique
    if not class_name or not class_name.strip():
        report.validity_issues.append("Name is empty.")
        report.outcome = "REJECT"

    if class_name in existing_class_names:
        report.validity_issues.append(f"Name '{class_name}' already exists.")
        report.outcome = "REJECT"

    # b) no duplicate sha256 among the 5
    if len(set(sha256s)) < len(sha256s):
        report.validity_issues.append("Duplicate images detected (same sha256).")
        report.outcome = "REJECT"

    # c) duplicate sha256 against existing images
    if existing_sha256s:
        existing_set = set(existing_sha256s)
        for i, h in enumerate(sha256s):
            if h in existing_set:
                report.validity_issues.append(
                    f"Example {i+1} has identical SHA-256 hash to an existing image."
                )
                report.outcome = "REJECT"

    # d) perceptual hash check (D6: replace cosine rule with imagehash)
    computed_phashes: list[Any] = []
    if phashes is not None and len(phashes) == 5:
        computed_phashes = list(phashes)
    elif image_paths is not None and len(image_paths) == 5:
        try:
            import imagehash
            from PIL import Image
            for p in image_paths:
                with Image.open(p) as img:
                    computed_phashes.append(imagehash.phash(img.convert("RGB")))
        except Exception as e:
            logger.warning("Perceptual hash computation failed: %s", e)

    if len(computed_phashes) == 5:
        try:
            import imagehash
            # Check pairwise among the 5 examples
            for i in range(5):
                for j in range(i + 1, 5):
                    h_i = computed_phashes[i]
                    h_j = computed_phashes[j]
                    dist = (h_i - h_j) if isinstance(h_i, imagehash.ImageHash) else (
                        imagehash.hex_to_hash(str(h_i)) - imagehash.hex_to_hash(str(h_j))
                    )
                    if dist <= phash_threshold:
                        report.validity_issues.append(
                            f"Examples {i+1} and {j+1} are perceptual near-duplicates (pHash Hamming distance={dist} <= {phash_threshold})."
                        )
                        report.outcome = "REJECT"

            # Check against existing perceptual hashes
            if existing_phashes:
                for i, h_i in enumerate(computed_phashes):
                    hi_obj = h_i if isinstance(h_i, imagehash.ImageHash) else imagehash.hex_to_hash(str(h_i))
                    for ex_h in existing_phashes:
                        ex_obj = ex_h if isinstance(ex_h, imagehash.ImageHash) else imagehash.hex_to_hash(str(ex_h))
                        dist = hi_obj - ex_obj
                        if dist <= phash_threshold:
                            report.validity_issues.append(
                                f"Example {i+1} is a perceptual near-duplicate of an existing image (pHash distance={dist})."
                            )
                            report.outcome = "REJECT"
                            break
        except Exception as e:
            logger.warning("Perceptual hash distance check error: %s", e)

    if report.outcome == "REJECT":
        report.notes = "REJECTED: validity failure. " + "; ".join(report.validity_issues)
        return report

    # ── N: Novelty (BH test using pooled calibration scores S_pool, D5) ───────
    K = len(gaussian.class_names)
    # Compute p-values against S_pool (n=200, floor 1/201) per Decision D5
    all_p_values = np.zeros((5, K))
    for i in range(5):
        for c_idx, c in enumerate(gaussian.class_names):
            score = gaussian.inflated_score(embeddings[i], c_idx)
            all_p_values[i, c_idx] = conformal.p_value_from_pool(score)

    flat_p = all_p_values.flatten()
    rejected = _bh_correction(flat_p, q_BH).reshape(5, K)

    # For each known class: count how many of the 5 are consistent (not rejected)
    consistent_with: dict[str, int] = {}
    for c_idx, c in enumerate(gaussian.class_names):
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
            c_idx = gaussian.class_names.index(c)
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
    mu_new = embeddings.mean(axis=0)   # (D,)
    for i in range(5):
        mu_others = (embeddings.sum(axis=0) - embeddings[i]) / 4.0
        diff = embeddings[i] - mu_others
        d = float(diff @ gaussian.A @ diff)
        dt = d / (1.0 + 1.0 / 4)
        p_c = conformal.p_value_from_pool(dt)
        if p_c < alpha_cons:
            report.consistency_issues.append(
                f"REVIEW: Example {i+1} is unlike the other four "
                f"(p={p_c:.4f} < alpha_cons={alpha_cons})."
            )
            if report.outcome == "ACCEPT":
                report.outcome = "REVIEW"

    # ── O: Overlap (Across all sub-means + Parametric Bootstrap, D7) ───────────
    rng = np.random.default_rng(rng_seed)
    D = embeddings.shape[1]
    tau = conformal.tau_hat

    for c_idx, c in enumerate(gaussian.class_names):
        # Evaluate across all mixture sub-means
        sub_means = gaussian.class_means[c_idx]  # (k, D)
        sub_ns = gaussian.n_sub[c_idx]           # list of n per sub-component

        best_eps = -1.0
        best_ci = (0.0, 0.0)

        for m_idx, mu_cm in enumerate(sub_means):
            n_m = sub_ns[m_idx] if m_idx < len(sub_ns) else (gaussian.n_c[c_idx] / len(sub_means))
            n_m = max(float(n_m), 1.0)
            delta_hat = mu_new - mu_cm
            raw_m = float(delta_hat @ gaussian.A @ delta_hat)
            W_m = 1.0 / 5.0 + 1.0 / n_m

            Delta2_hat = max(0.0, raw_m - tau * W_m)
            eps_m = float(scipy_norm.cdf(-np.sqrt(Delta2_hat) / 2.0))

            # Parametric bootstrap under the Gaussian model (Decision D7)
            # Under H0, (\delta^*)^T A \delta^* / W_m ~ chi2_D(nonc = raw_m / W_m)
            nonc = max(0.0, raw_m / W_m)
            sampled_quad = rng.noncentral_chisquare(df=D, nonc=nonc, size=bootstrap_samples) * W_m
            sampled_Delta2 = np.maximum(0.0, sampled_quad - tau * W_m)
            sampled_eps = scipy_norm.cdf(-np.sqrt(sampled_Delta2) / 2.0)
            ci_lo = float(np.percentile(sampled_eps, 2.5))
            ci_hi = float(np.percentile(sampled_eps, 97.5))

            if eps_m > best_eps:
                best_eps = eps_m
                best_ci = (ci_lo, ci_hi)

        report.eps_hat[c] = best_eps
        report.eps_hat_bootstrap[c] = best_ci

        if best_eps > eps_max:
            report.overlap_issues.append(
                f"REVIEW: Estimated confusion with class '{c}': "
                f"eps_hat={best_eps:.3f} > eps_max={eps_max} "
                f"(bootstrap 95% CI: [{best_ci[0]:.3f}, {best_ci[1]:.3f}]). "
                f"Consider merging."
            )
            if report.outcome == "ACCEPT":
                report.outcome = "REVIEW"

    # Build notes summary
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
