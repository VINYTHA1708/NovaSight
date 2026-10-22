# NovaSight Protocol v1.0 Deviations Log

**Document Version**: 1.1  
**Date**: 22 October 2026  
**Reference Protocol**: `docs/RESEARCH_SPEC.md` (v1.0) -> `docs/RESEARCH_SPEC_v1.1.md` (v1.1)  
**Purpose**: Authoritative audit record of all deviations, implementation discrepancies, and empirical findings observed in the Protocol v1.0 implementation, with exact verbatim file:line code excerpts and official dispositions approved under Protocol v1.1.

---

## 1. Summary Matrix of Protocol v1.0 Audit Items

Items are indexed by owner audit IDs: **L1–L8** (known audit facts) and **§5 items 1–17** (detailed technical audit findings). Each item is classified as either a **Deviation** (departure from specification) or a **Finding** (empirical measurement, dataset property, or diagnostic observation).

| # | Audit ID | Type | Category | Summary Description | File:Line Evidence | Disposition (Protocol v1.1) |
|---|---|---|---|---|---|---|
| **1** | L1 / §5.1 | Deviation | Model Selection Leakage | `scripts/select_model.py` sub-sampled 40 slices from `unk_test[:40]` during candidate model selection, violating clean split boundaries. | `scripts/select_model.py:92-96` | **HISTORICAL_NONCOMPLIANT (D1)**. Selection rewritten in `scripts/rehearse.py` (E5) using nested LOCO on `sel_cal`/`sel_eval` without accessing `unk_test`. |
| **2** | L2 / §5.2 | Deviation | Out-of-Order Execution | E1 primary run was evaluated on test data in commit `365bf7e` *before* `scripts/select_model.py` was committed in `3fd4bbc`. | `git log --oneline` (`365bf7e` vs `3fd4bbc`) | **HISTORICAL_NONCOMPLIANT (D1)**. Primary v1.0 run classified as exploratory; never used as confirmatory evidence. |
| **3** | L3 / §5.3 | Deviation / Finding | Duplicate Pool Straddling | Byte-identical duplicate images `Pa_101.bmp` and `Pa_105.bmp` straddle pools in 18 of 30 v1 manifests (`Pa_train` vs `Pa_test` in `H=Sc_s=0`); `splits.py` checked disjointness by path, not content hash. | `src/data/splits.py:276-299`; `data/manifests/H=Sc_s=0.json` | **Splits v2 (D2)**. v1 manifests kept untouched for auditability. Splits v2 generated in `data/manifests/v2/` treating duplicate pairs as an atomic hash-disjoint unit. |
| **4** | L4 / §5.4 | Deviation | Unenforced Pool Guard | Data-access guard `PoolGuard` was never called by any dataset loading or pipeline execution script. | `src/data/guard.py:49`; `src/data/splits.py` (0 calls to `check_pool_access`) | **Task E3**. Enforce `check_pool_access` in all dataset loaders, caching scripts, and evaluation entry points. |
| **5** | L5 / §5.5a | Deviation | Missing B1 Checkpoint | Fine-tuned B1 weights were never persisted to disk; `make_extractor("B1")` returned standard ImageNet weights; held-out B1 extraction lacked state_dict hash assertion. | `src/models/backbones.py:420-425`; `scripts/run_finetune.py`; `scripts/_extract_b1_heldout.py` | **Task E4**. Save B1 weights to `models/` with SHA-256; `make_extractor("B1")` loads saved checkpoint or fails loudly. |
| **6** | §5.5b | Finding | High Known Ambiguity Rate | At $\alpha=0.05$, the v1.0 primary run produced 172 ambiguous predictions out of 200 known test images ($86.0\%$), and 3 empty sets ($1.5\%$), yielding an operator routing rate of $175/200 = 87.5\%$. | `reports/results_H=Sc_s=0_B0_L4.json`; `reports/confusion_matrix_H=Sc_s=0_B0_L4.json` | **Standardized Terminology (Part B)**. Distinguish `empty_rate` from `ambiguous_rate` and `operator_routing_rate`. Report exploratory $\hat{c}$ acceptance (D4). |
| **7** | L6 / §5.6a | Deviation | Conflated Coverage vs Recall | `scripts/evaluate.py` reported `new_class_coverage` $P(p_{\text{new}} > \alpha)$ under the label `"recall_h1"`; singleton recall $\{new\}$ was unmeasured; `bootstrap_ci` was an uninvoked placeholder. | `scripts/evaluate.py:142-156, 283-288` | **Task E6 & E7**. Implement Part B terminology (`new_class_coverage` vs `new_class_recall` vs `new_class_FRR`); evaluate both H1 and H1-s; wire in 1,000-resample image bootstrap CIs. |
| **8** | §5.6b | Finding | Low New-Class Coverage | At $\alpha=0.05$, 10 five-shot enrollment draws achieved mean H1 coverage of only $0.1296$ (approx $13\%$), implying high new-class FRR $\approx 87\%$ ($1 - 0.13$). | `reports/results_H=Sc_s=0_B0_L4.json`; `scripts/evaluate.py:299-300` | **Documented in README/DEV_LOG**. Formalized in Hypothesis H1 / H1-s testing across all classes in Gate 6. |
| **9** | L7 / §5.7a | Deviation | Metric Definition Mismatch | `scripts/evaluate.py` counted only empty sets as rejected ($\Gamma = \emptyset$), while `src/eval/metrics.py` counted non-singletons ($\Gamma \ne \{c\}$) as rejected in the confusion matrix. | `scripts/evaluate.py:229-234`; `src/eval/metrics.py:254-257, 287` | **Standardized Terminology (Part B)**. Distinguish `empty_rate`, `ambiguous_rate`, `known_FRR`, and `operator_routing_rate`. Display `"ambiguous"` as an independent column in confusion matrices. |
| **10** | §5.7b | Deviation | Gate N-Check Power Deficiency | In `run_gate`, BH compares the k-th smallest p-value with $q \cdot k / m = 0.002k$ ($q=0.05, m=25$). With a per-class floor of $1/41 \approx 0.0244$, rejection requires at least 13 of the 25 p-values at the floor, so power is very low (not zero). Under D5 (floor $1/201 \approx 0.005$) at least 3 floor-level p-values are still needed. | `src/models/gate.py:180-208` | **Decision D5**. Novelty check N uses pooled-calibration p-values against $S_{\text{pool}}$ ($n=200$, floor $1/201 \approx 0.005$). |
| **11** | §5.7c | Deviation | Gate Validity Cosine Rule | Check V rejected exemplars with pairwise cosine similarity $\ge 0.95$; exemplars can exceed cosine 0.95 (post-ReLU pooled features are non-negative); not measured on NEU-CLS. | `src/models/gate.py:151-174` | **Decision D6**. Remove the $\text{cosine} \ge 0.95$ rule; enforce exact SHA-256 and perceptual-hash (imagehash) checks. |
| **12** | §5.7d | Deviation | Gate Overlap Range & Sub-mean Flaw | Check O used only the first sub-mean `class_means[c_idx][0]` and substituted an ad-hoc analytic $\pm 1$ SE formula rather than a parametric bootstrap range. | `src/models/gate.py:241-264` | **Decision D7**. Implement the parametric bootstrap for the $\hat{\epsilon}$ range under the Gaussian model as specified; evaluate proper mixture sub-means. |
| **13** | L8 / §5.8 | Deviation | App Gate Policy Inversion | Gradio application `app.py` enrolled new classes even when `gate_report.outcome == "REJECT"`; featured an unconstrained alpha slider, 4 tabs (omitting Operator Queue), and global mutable state in `state/`. | `app.py:117, 295-303` | **Task E12 (D.8)**. In the application, `REJECT` strictly blocks enrollment; `REVIEW` requires written operator notes; isolated per-session sandbox implemented. |
| **14** | §5.9 | Deviation | Unsupported Bootstrap CIs | `README.md:79-80` reported 95% bootstrap confidence intervals for T1 accuracy and T2 AUROC that no committed output file supported. | `README.md:79-80`; `reports/results_H=Sc_s=0_B0_L4.json` | **Decision D11**. Removed unsupported CIs from `README.md`; re-generate CIs strictly from committed code outputs in Gate 5/6. |
| **15** | §5.10 | Deviation | Specification Incompleteness | `docs/RESEARCH_SPEC.md` claimed to be verbatim Parts 4–9 but omitted formal sections D.1–D.8 of the master prompt. | `docs/RESEARCH_SPEC.md` (319 lines vs full master prompt) | **Restored in v1.1**. `docs/RESEARCH_SPEC_v1.1.md` restores master specification; v1.0 spec preserved with clear notice. |
| **16** | §5.11 | Finding | Line-Ending & Config Hash Mismatch | `metadata.json:config_sha256` (`a6342c18...`) reflects Windows working copy on disk with line 313 ending in CRLF, while Git blob has LF (`5364ceee...`). | `artifacts/base/metadata.json:33`; `configs/default.yaml:313` | **Gate 1 Part A / Task E1**. Line 313 had `\r\n` (offset 10,898), 10,900 B CRLF vs 10,899 B LF. Add `.gitattributes` (`eol=lf`), introduce canonical hashing. |
| **17** | §5.12 | Finding | Confusion Matrix Hash Discrepancy | Recorded in POST_IMPLEMENTATION_AUDIT.md as SHA-256 `18f844b2...` for `confusion_matrix_H=Sc_s=0_B0_L4.json`, while actual file hashes to `3b6d4154...` (CRLF) / `b9483a2c...` (LF); origin unknown; not reproducible from any committed file (not in README.md). | `reports/confusion_matrix_H=Sc_s=0_B0_L4.json`; `reports/POST_IMPLEMENTATION_AUDIT.md` | **Corrected in Gate 2**. Hash discrepancy corrected in documentation; line-ending dependence documented; canonical hashing pending E1. |
| **18** | §5.13 | Finding | Result File Line-Ending Hash | Result file `results_H=Sc_s=0_B0_L4.json` hashes to `e2b3ef73...` on Windows disk (CRLF), but hashes to `d44bbb67...` on committed Git blob (LF). | `reports/results_H=Sc_s=0_B0_L4.json`; `reports/run_registry.csv` | **Documented in run_registry.csv**. CRLF and LF hashes both recorded in registry notes; canonical hashing pending E1. |
| **19** | §5.14 | Finding / Deviation | Selection Scope & Candidate Failure | `scripts/select_model.py` evaluated only L4 features; never evaluated B2, L234, or option (b); all evaluated candidates had selection FRR of 0.965–1.0 on `sel_eval`. | `scripts/select_model.py:175-180`; `reports/selection_result.json` | **Task E5**. Rehearsal protocol evaluates all pre-declared candidates on permitted rehearsal pools per D.2. |
| **20** | §5.15 | Deviation | Selection Objective Divergence & Fitting on Rehearsal Cal | `scripts/select_model.py` fit Gaussian on `sel_cal` rather than `train`; used scalar composite `AUROC * (1 - FRR)` and varied $\alpha$ rather than fixed $\alpha=0.05$ with Part 5.3 hierarchy. | `scripts/select_model.py:98-101, 139, 192` | **Decision D3 & Task E5**. Fix $\alpha = 0.05$ for all operational decisions; fit on `train`; calibrate on `sel_cal`; implement Part 5.3 hierarchical tie-breaking rule. |
| **21** | §5.16 | Finding | Synthetic-Only Test Regime | S1 tests use the real NEU-CLS index and manifests; S2 uses real backbone weights; S4/S6 use synthetic, well-separated classes. No test covers the realistic regime (overlapping classes, D=512, n=120) or end-to-end rehearsal/leakage. | `tests/test_s1.py`; `tests/test_s2.py`; `tests/test_s4.py`; `tests/test_s6.py` | **Task E8 & Part F**. Add data-guard enforcement tests, leakage detection tests, and verify pipeline execution against real caches. |
| **22** | §5.17 | Deviation | Build Artifact Provenance Flaws & Cache Key Omission | `scripts/build_artifact.py` recorded absolute filesystem paths and no pool hashes in `provenance.json`; `cache_features.py` omitted weight/preprocessing hashes from cache filenames; `backbones.py` ignored `cache_dir` in B2 hub load. | `scripts/build_artifact.py:180-191`; `scripts/cache_features.py:57-62`; `src/models/backbones.py:340-356` | **Task E1, E4, E10**. Provenance records relative paths and pool SHA-256 hashes; feature cache paths include weight/preprocessing hashes; B2 hub loading respects `cache_dir`. |

---

## 2. Detailed Technical Evidence & Verbatim Code Excerpts

### 2.1 Model Selection Test Pool Leakage (L1 / §5.1)
- **Prescribed Boundary**: Section 5.1 of `docs/RESEARCH_SPEC.md` reserves `unk_test` exclusively for final test evaluation:
  - `sel_cal`: rehearsal calibration + rehearsal enrollment draws + injection sources
  - `sel_eval`: rehearsal evaluation, gate tuning, injection experiments
  - `unk_test`: unknown-detection metrics ONLY
- **Actual Code (`scripts/select_model.py:92-96`)**:
```python
    # Load held-out sel_eval features (use unk_test as proxy for sel_eval unknown)
    unk_path = get_cache_path(backbone, feature, held_out, seed, "unk_test", held_out)
    if not unk_path.exists():
        return None
    unk_feats = np.load(unk_path)["features"][:40]  # use 40 for sel_eval (same size as sel_eval)
```
- **Impact**: Final test data was directly utilized to rank candidate architectures and mixture components, contaminating the candidate selection process.
- **Disposition**: Run classified as `HISTORICAL_NONCOMPLIANT`. Protocol v1.1 implements rehearsal strictly via nested leave-one-class-out across known classes in `sel_cal` and `sel_eval`.

### 2.2 Content-Identical Duplicate Image Straddling (L3 / §5.3)
- **Prescribed Boundary**: Section 5.1 specifies: "Stratified, seeded, hash-disjoint pools".
- **Actual Code (`src/data/splits.py:276-299`)**:
```python
def _assert_disjoint(pools: dict[str, list[dict[str, Any]]]) -> None:
    """Assert that no file PATH appears in more than one pool.

    Note: Two different files with the same sha256 (identical content but
    different filenames) may legitimately exist in different pools.
    The check uses file paths as the uniqueness key, not sha256.
    Actual duplicate files (same path) would be a logic error.

    Args:
        pools: Dict of pool_name -> list of records.

    Raises:
        ValueError: If any file path appears in more than one pool.
    """
    seen: dict[str, str] = {}  # path -> pool_name
    for pool_name, records in pools.items():
        for rec in records:
            path = rec["path"]
            if path in seen:
                raise ValueError(
                    f"File {path} appears in pools "
                    f"{seen[path]} and {pool_name} -- this is a logic error"
                )
            seen[path] = pool_name
```
- **Impact**: Disjointness was checked by file path rather than image content hash. The identical pair `Pa_101.bmp` ≡ `Pa_105.bmp` (SHA-256 `6833002b8b74289467d30f3050c904944b0fa56e30b3a36417f73db1c1a96752`) straddled pools in 18 of 30 manifests. In `H=Sc_s=0.json`, `Pa_101.bmp` is in `Pa_train` while `Pa_105.bmp` is in `Pa_test`.
- **Disposition**: Under Decision D2, Splits v2 (`data/manifests/v2/`) will treat duplicate pairs as an atomic unit assigned to the same pool, ensuring genuine hash disjointness. Historical v1 manifests are preserved untouched.

### 2.3 Evaluator Metric Conflation & Unused Bootstrap Placeholder (L6 / §5.6)
- **Prescribed Protocol**: Section 6.3 requires post-enrollment singleton recall $\{new\}$, new-class coverage, and bootstrap confidence intervals.
- **Actual Code (`scripts/evaluate.py:283-288`)**:
```python
        for f in new_test_feats:
            score_h1 = cls.inflated_score(f, gaussian.A, conformal.tau_hat)
            p_h1 = conformal.p_value_from_pool(score_h1)
            h1_recalls.append(p_h1 > alpha)

        recall_h1 = float(np.mean(h1_recalls))
```
  Note also that `scripts/evaluate.py:142-156` defines `bootstrap_ci()`, which is never invoked anywhere in the evaluation routine.
- **Impact**: Conformal coverage $P(p_{\text{new}} > \alpha)$ was misreported as recall, singleton recall $\{new\}$ was unmeasured, and confidence intervals in `README.md` had no supporting output file.
- **Disposition**: Under Decision D11, unsupported CIs are removed from `README.md`. Evaluator v1.1 (`scripts/evaluate_v11.py`) implements Part B terminology (`new_class_coverage` vs `new_class_recall` vs `new_class_FRR`) and generates bootstrap CIs from code.

### 2.4 Model Selection Gaussian Fitting on Rehearsal Calibration (L1 / §5.15)
- **Prescribed Protocol**: Section 5.1 specifies:
  - `train`: B1 weights, Gaussian means/covariance, mixture sub-means, cross-fitted weights, linear-probe baseline
  - `sel_cal`: rehearsal calibration, rehearsal enrollment draws, injection sources
- **Actual Code (`scripts/select_model.py:98-104`)**:
```python
    # Fit Gaussian on sel_cal
    train_sel = {PREFIX_TO_CLASS[p]: sel_cal_feats[p] for p in known_prefixes if p in sel_cal_feats}
    try:
        gaussian = fit_gaussian(train_sel, k=k)
    except Exception as e:
        logger.warning("Gaussian fit failed for (%s/%s, k=%d): %s", backbone, feature, k, e)
        return None
```
- **Impact**: Gaussians in model selection were fitted on `sel_cal` (30 samples/class) rather than `train` (120 samples/class), distorting the covariance estimate and candidate ranking.
- **Disposition**: Task E5 implements candidate rehearsal fitting strictly on `train` of known classes, calibrating on `sel_cal`, and evaluating on `sel_eval`.

### 2.5 Gate Validity Check Cosine Threshold (§5.7c)
- **Prescribed Protocol**: Section 4.7 specifies validity checks: decodable, size OK, not near-constant, no duplicate sha256, no near-duplicate embeddings.
- **Actual Code (`src/models/gate.py:151-161`)**:
```python
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
```
- **Impact**: Legitimate exemplars can exceed cosine 0.95 (post-ReLU pooled features are non-negative); not measured on NEU-CLS, risking unwarranted rejections during enrollment.
- **Disposition**: Decision D6 supersedes this rule; exact SHA-256 and perceptual-hash (imagehash) checks are enforced instead.

### 2.6 Gate Novelty Check Power Deficiency (§5.7b)
- **Prescribed Protocol**: Section 4.7 specifies Benjamini-Hochberg across all $5 \times K$ tests at level $q_{\text{BH}}$.
- **Actual Code (`src/models/gate.py:180-189`)**:
```python
    K = len(gaussian.class_names)
    # Compute p-values for each of the 5 examples vs each known class (5*K tests)
    all_p_values = np.zeros((5, K))
    for i in range(5):
        for c_idx, c in enumerate(gaussian.class_names):
            score = gaussian.inflated_score(embeddings[i], c_idx)
            all_p_values[i, c_idx] = conformal.p_value(score, c)

    flat_p = all_p_values.flatten()
    rejected = _bh_correction(flat_p, q_BH).reshape(5, K)
```
- **Impact**: BH compares the k-th smallest p-value with $q \cdot k / m = 0.002k$ ($q=0.05, m=25$). With a per-class floor of $1/41 \approx 0.0244$, rejection requires at least 13 of the 25 p-values at the floor, so power is very low (not zero). Under D5 (floor $1/201 \approx 0.005$) at least 3 floor-level p-values are still needed.
- **Disposition**: Decision D5 supersedes this; check N uses pooled calibration scores $S_{\text{pool}}$ ($n=200$, floor $1/201 \approx 0.005$).

### 2.7 Gate Overlap Analytic Range & First Sub-Mean Flaw (§5.7d)
- **Prescribed Protocol**: Section 4.7 specifies estimating $\hat{\epsilon}_c$ and reporting a parametric-bootstrap range under the Gaussian model.
- **Actual Code (`src/models/gate.py:241-264`)**:
```python
    for c_idx, c in enumerate(gaussian.class_names):
        mu_c = gaussian.class_means[c_idx][0]   # use first sub-mean for approximation
        n_c = gaussian.n_c[c_idx]
        delta_hat = mu_new - mu_c
        raw = float(delta_hat @ gaussian.A @ delta_hat)
        tau = conformal.tau_hat
        # Delta2_hat = max(0, delta^T A delta - tau_hat * (1/5 + 1/n_c))
        Delta2_hat = max(0.0, raw - tau * (1.0 / 5 + 1.0 / n_c))
        eps_hat = float(scipy_norm.cdf(-np.sqrt(Delta2_hat) / 2.0))
        report.eps_hat[c] = eps_hat

        # Parametric bootstrap range (±1 SE approximation)
        # SE of Delta2_hat is not easy to compute; use ±1 sd of raw term
        # under Gaussian model: Var(delta^T A delta) ≈ 2 * tr((A Sigma)^2) * (1/5 + 1/n_c)^2
        A = gaussian.A
        Sigma_hat = np.linalg.inv(A)
        ASigma = A @ Sigma_hat
        var_raw = 2.0 * np.trace(ASigma @ ASigma) * (1.0 / 5 + 1.0 / n_c) ** 2
        se_raw = float(np.sqrt(max(var_raw, 0.0)))
        Delta2_lo = max(0.0, raw - se_raw - tau * (1.0 / 5 + 1.0 / n_c))
        Delta2_hi = max(0.0, raw + se_raw - tau * (1.0 / 5 + 1.0 / n_c))
        eps_lo = float(scipy_norm.cdf(-np.sqrt(Delta2_hi) / 2.0))
        eps_hi = float(scipy_norm.cdf(-np.sqrt(Delta2_lo) / 2.0))
        report.eps_hat_bootstrap[c] = (eps_lo, eps_hi)
```
- **Impact**: Overlap check used only the first sub-mean (`gaussian.class_means[c_idx][0]`) rather than accounting for multiple mixture sub-means, and substituted an ungrounded $\pm 1$ SE heuristic for the required parametric bootstrap.
- **Disposition**: Decision D7 implements the parametric bootstrap for the $\hat{\epsilon}$ range under the Gaussian model.

### 2.8 Application Gate Policy Inversion (L8 / §5.8)
- **Prescribed Protocol**: Section 4.7 / 8: "In the APP, REJECT blocks enrollment and REVIEW needs a written acknowledgement... (In EVALUATION, the gate is recorded but never blocks enrollment)".
- **Actual Code (`app.py:295-303`)**:
```python
    # Enroll regardless of gate outcome [APPROVED-POLICY]
    mem_fresh = get_memory()
    mem_fresh.enroll(
        name=class_name,
        embeddings=embeddings_arr,
        sha256s=sha256s,
        gate_report=gate_report.to_dict(),
        operator_id="operator",
    )
```
- **Impact**: The UI enrolled classes regardless of whether the gate returned `REJECT`, disabling the safety gate in interactive operation.
- **Disposition**: Under Task E12, the application will enforce `REJECT` blocking and require written operator notes on `REVIEW`.

### 2.9 Provenance Logging Flaws & Cache Key Omission (§5.17)
- **Actual Code (`scripts/build_artifact.py:180-191`)**:
```python
    prov_path = REPO_ROOT / "artifacts" / "provenance.json"
    write_provenance(
        prov_path,
        stage="S5_build_artifact",
        inputs={
            "train_cache": str(get_cache_path(args.backbone, args.feature, held_out, split_seed, "train", "Cr")),
            "cal_cache": str(get_cache_path(args.backbone, args.feature, held_out, split_seed, "cal", "Cr")),
        },
        outputs=artifact_hashes,
        notes=f"Primary run: held_out={held_out}, seed={split_seed}, backbone={args.backbone}, "
              f"feature={args.feature}, k={args.k}, elapsed={elapsed:.1f}s",
    )
```
- **Actual Code (`scripts/cache_features.py:57-62`)**:
```python
    cfg = get_config()
    cache_root = REPO_ROOT / cfg["paths"]["feature_cache"]
    run_key = f"H={held_out}_s={split_seed}"
    pool_key = f"{prefix}_{pool_name}" if prefix else pool_name
    name = f"{backbone_id}_{feature}_{run_key}_{pool_key}.npz"
    return cache_root / name
```
- **Actual Code (`src/models/backbones.py:340-356`)**:
```python
        if cache_dir is None:
            cache_dir = REPO_ROOT / cfg["backbones"]["B2"]["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)

        hub_repo = cfg["backbones"]["B2"]["hub_repo"]
        hub_model = cfg["backbones"]["B2"]["hub_model"]

        logger.info("Loading DINOv2 (%s/%s)...", hub_repo, hub_model)
        t0 = time.time()
        try:
            self._model = torch.hub.load(
                hub_repo,
                hub_model,
                verbose=False,
                source="github",
            )
```
- **Impact**: Provenance files contained machine-specific absolute paths and omitted manifest SHA-256 hashes; feature caches did not invalidate upon model weight changes; DINOv2 hub loading bypassed local `cache_dir`.
- **Disposition**: Tasks E1, E4, E10 ensure portable relative paths, input pool hashes, weight-keyed cache filenames, and compliant backbone caching.

### 2.10 Hash & Line-Ending Investigation (§5.11, §5.12, §5.13)
- **`config_sha256` Root Cause**:
  - Committed blob (`HEAD:configs/default.yaml`): 10,899 bytes, pure LF (313 LF, 0 CR), SHA-256 = `5364ceee4565d4d470ad03a3be2ff6280b7467d561edb64b2a2c1b347224945f`.
  - Windows working file (`configs/default.yaml`): 10,900 bytes, mixed line endings (313 LF, 1 CR at line 313 offset 10,898), SHA-256 = `a6342c1874be611a535ba025644554f1df2622af46762ac5c6b6c05083d0b0a3`.
  - Explicit transformation: `blob[:-1] + b'\r\n'` reproduces `a6342c18...` bit-for-bit.
  - `git ls-files --eol` reports `i/lf w/mixed attr/`. `git hash-object` matches HEAD (`e8290499...`) because `core.autocrlf=true` filters the lone `\r`.
- **Confusion Matrix Hash**:
  - `reports/confusion_matrix_H=Sc_s=0_B0_L4.json` actual hash: `3b6d4154...` (CRLF) / `b9483a2c...` (LF).
  - The hash `18f844b2...` recorded in `POST_IMPLEMENTATION_AUDIT.md` has origin unknown; not reproducible from any committed file (verified absent from `README.md`).
- **Result File Hash**:
  - `reports/results_H=Sc_s=0_B0_L4.json` hashes to `e2b3ef73...` (CRLF) and `d44bbb67...` (LF). Both are documented in `reports/run_registry.csv`.
- **Disposition**: Task E1 will add `.gitattributes` (`eol=lf`) and canonical line-ending-independent hashing.
