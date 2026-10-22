# NovaSight Research Specification — Protocol v1.1

**Version**: 1.1 (Condensed restatement of the final master prompt Parts 4–9; not verbatim)  
**Date**: 22 October 2026  
**Supercedes**: `docs/RESEARCH_SPEC.md` (Protocol v1.0)  
**Status**: Authoritative research protocol. Frozen prior to any new empirical runs.  
**Precedence**: Where Part 9 (Clarifications D1–D11) conflicts with Parts 4–8, Part 9 governs.  

---

## PART 4 — METHOD SPECIFICATION

### 4.1 Preprocessing
Load as grayscale (convert RGB uploads to L) -> resize to 224 (bilinear) -> repeat to 3 channels -> backbone-specific normalisation (ImageNet mean/std for ResNet; DINOv2's own normalisation for B2).
Input sanity checks (heuristic, not calibrated) — APPLICATION INPUT ONLY: undecodable file, min side < 32 px, near-constant image (grayscale pixel std < 1.0 on the 0–255 scale) [APPROVED-POLICY] -> reject with a specific message.
These checks are NEVER applied to evaluation data and never silently drop images. In S1, scan all 1,800 NEU images with the same checks and write `reports/sanity_scan.json` (count + filenames of any image that would trip). If any image trips, report it in README and DEV_LOG; it stays in its evaluation pool.

### 4.2 Backbone candidates (all must be implemented and compared; no winner may be assumed)
- **B0**: torchvision ResNet-18, `IMAGENET1K_V1` weights, frozen.
- **B1**: ResNet-18 from the same weights, fine-tuned with a linear head (5-way for final models, 4-way in nested rehearsal) and cross-entropy on known-class TRAIN data only; early stopping on EARLY-STOP data only.
  - *[APPROVED-POLICY] single fixed configuration (no hyperparameter grid)*: only layer4 unfrozen (stem–layer3 frozen); backbone LR 1e-4; head LR 1e-3; AdamW, weight decay 1e-4; batch 32; max 15 epochs; patience 3 on early-stop cross-entropy loss (restore best epoch); augmentation = horizontal and vertical flips + brightness/contrast jitter of $\pm 0.1$; model-initialisation seed from config (head init and data order). Document in README that this is a fixed engineering choice, not an optimised configuration. The same configuration is used for every nested rehearsal fine-tune and every final fine-tune.
- **B2**: DINOv2 ViT-S/14 (`torch.hub facebookresearch/dinov2`, `"dinov2_vits14"`), frozen. This needs network access on first run; cache the weights and record the hash. Verify and record its licence in README. If the weights cannot be obtained, report B2 as "not run". NEVER substitute another model under the B2 name. (See Decision D9).

Every backbone: record the weight source, sha256 and init seed. Load from the local cache when present; fail with a clear message, not a silent fallback, when absent. Everything must run on CPU; use a GPU only if one is present, and never require it.

### 4.3 Feature options (compared in rehearsal)
- ResNet (B0/B1): L4 = layer4 global-average-pooled ($D=512$); L234 = concatenation of GAP(layer2) 128 + GAP(layer3) 256 + GAP(layer4) 512 = 896, each block z-scored with TRAIN statistics.
- DINOv2 (B2): CLS token ($D=384$); CLS concatenated with mean patch token ($D=768$).
- Combination options for multi-block features:
  - (a) one Gaussian on the concatenated vector (default);
  - (b) per-block Gaussian scores combined by logistic regression whose weights are learned by 5-fold CROSS-FITTING WITHIN TRAIN ONLY: fit per-block Gaussians on 4 folds, score the held-out fold (known positives); create pseudo-unknown negatives by leaving one class out within the out-of-fold scores; fit the logistic regression on these out-of-fold scores only. The weights must be a function of TRAIN alone (unit test: perturbing any other pool leaves the weights bit-identical).
Do NOT reduce dimensionality with top-variance PCA. Low-variance directions matter for novelty (Rippel et al. 2020).

### 4.4 Class-conditional Gaussian model (shared covariance)
Working ASSUMPTION (never present it as a fact about NEU):
$$f \mid Y=c \sim \mathcal{N}(\mu_c, \Sigma), \quad \Sigma \text{ shared across all classes.}$$
Estimate $\mu_c = \text{class mean of TRAIN features}$ ($n_c = 120$ per known class). Residuals $R = f_i - \hat{\mu}_{y_i}$. $\hat{\Sigma} = \text{Ledoit-Wolf shrinkage estimate computed from } R \text{ only}$ (`assume_centered=True` on residuals). $A = \hat{\Sigma}^{-1}$, computed stably; assert positive definiteness; log the condition number and the shrinkage intensity.
- Score: $d_c(f) = (f - \mu_c)^T A (f - \mu_c)$.
- Inflated score: $\tilde{d}_c(f) = d_c(f) / (1 + 1/n_c)$.
- Closed-set label: $\hat{c} = \arg\min_c \tilde{d}_c(f)$.
- Mixture variant [APPROVED-POLICY]: k-means (seeded) sub-means per known class fitted on TRAIN only, $k \in \{1,2,3\}$ ($k=1$ is the plain Gaussian), sharing the same $A$; class score = min over sub-means, each sub-mean inflated by $(1 + 1/n_{\text{sub}})$. $k$ is a candidate dimension selected ONLY by the rehearsal rule in 5.3. Enrolled classes always use a single mean.

### 4.5 Conformal decision (class-conditional / Mondrian)
For each known class $c$:
$$S_c = \{ \tilde{d}_c(x_i) : x_i \in \text{CAL}, y_i = c \} \quad (40 \text{ scores per class}).$$
$$p_c(x) = \frac{1 + \#\{s \in S_c : s \ge \tilde{d}_c(x)\}}{|S_c| + 1}.$$
Prediction set $\Gamma_\alpha(x) = \{ c : p_c(x) > \alpha \}$.
Output rule:
- empty set $\Gamma_\alpha(x) = \emptyset$ -> "unknown, send to operator"
- singleton set $\{c\}$ -> class $c$
- 2+ classes -> "ambiguous between ..." -> operator review (also show $\hat{c}$).

Primary novelty score for ALL AUROC / AUPR / FPR@95 / OSCR computation, for both rehearsal selection and final reporting [APPROVED-POLICY]:
$$N(x) = \min_c \tilde{d}_c(x)$$
over all classes currently in memory. Conformal p-values are NOT used for AUROC (their $1/41$ granularity causes ties); they drive only the decision rule (prediction sets, unknown/ambiguous outputs, FRR at alpha, coverage).
Default $\alpha = 0.05$; full grid $\{0.01, 0.02, 0.05, 0.10, 0.20\}$.
INVARIANT (must be unit-tested): adding or removing an enrolled class never changes any known class's p-values (bit-identical).
$$S_{\text{pool}} = \bigcup_{c} S_c \quad (200 \text{ scores}); \quad \hat{\tau} = \text{mean}(S_{\text{pool}}).$$

### 4.6 Guarantee statement
- **G1 (known classes)**: If (A1) a test image of class $c$ and class $c$'s CAL images are exchangeable given $Y=c$, and (A2) the score function does not depend on CAL or test images, then $P(p_c(X) \le \alpha \mid Y=c) \le \alpha$. Hence $P(\text{rejected as unknown} \mid Y=c) \le \alpha$, and $P(\text{wrong single label} \mid Y=c) \le \alpha$. G1 holds on average over calibration sets. For a fixed CAL set, coverage $\sim \text{Beta}(n+1-l, l)$ with $l = \lfloor(n+1)\alpha\rfloor$; at $n=40, \alpha=0.05$ this is $\text{Beta}(39,2)$ (mean approx 0.951, sd approx 0.033). Report this spread.
- **NOT guaranteed**: (N1) unknown-detection power; (N2) prediction-set size; (N3) behaviour under distribution shift or non-random selection; (N4) anything about enrolled classes.

### 4.7 Five-shot enrollment
Inputs: exactly 5 images + name. Embed with the deployed backbone.
Checks:
- **V (validity)** (superseded by D6): decodable, size OK, not near-constant, no duplicate sha256, no near-duplicate embeddings, name non-empty and unique -> REJECT on failure.
- **N (novelty)** (superseded by D5): $p_c(f_i)$ for every example $i$ and known class $c$; Benjamini-Hochberg across all $5 \times K$ tests at level $q_{\text{BH}}$. If $\ge 3$ of 5 examples are consistent with the SAME known class $c$ -> REJECT ("these look like c; label them as c?"). If 1–2 examples are -> REVIEW, naming them.
- **C (consistency)**: for each $i$, the distance from $f_i$ to the mean of the other four, inflated by $(1 + 1/4)$, converted to a p-value against $S_{\text{pool}}$. Any $p < \alpha_{\text{cons}}$ -> REVIEW ("example $i$ is unlike the other four").
- **O (overlap)**: $\hat{\delta} = \mu_{\text{new}} - \mu_c$; $\hat{\Delta}^2 = \max(0, \hat{\delta}^T A \hat{\delta} - \hat{\tau}(1/5 + 1/n_c))$; $\hat{\epsilon}_c = \Phi(-\sqrt{\hat{\Delta}^2}/2)$, a plug-in approximation that treats $\hat{\Sigma}$ as $\Sigma$. Report a parametric-bootstrap range under the Gaussian model (see Decision D7). Any $\hat{\epsilon}_c > \epsilon_{\text{max}}$ -> REVIEW ("estimated ~X% confusion with $c$; consider merge").

Outcomes:
- `ACCEPT`: class enrolled cleanly.
- `REVIEW`: operator must acknowledge each warning with a written note, or replace the flagged examples.
- `REJECT`: cannot be overridden.
*(In EVALUATION, the gate outcome is recorded but never blocks enrollment; see 6.1.)*

On acceptance: $\mu_{\text{new}} = \text{mean of the 5 embeddings}$; covariance = shared $A$; $n = 5$, so inflation is 1.2.
- **H1 (plain inheritance)**: $p_{\text{new}}(x)$ uses $S_{\text{pool}}$ with $\tilde{d}_{\text{new}} = d_{\text{new}} / 1.2$.
- **H1-s (scale-adaptive)**: $\hat{r} = (\text{mean of 5 leave-one-out distances, each inflated by } 1 + 1/4) / \hat{\tau}$; score $= \tilde{d}_{\text{new}} / \hat{r}$, compared against $S_{\text{pool}}$.

The deployed app uses the variant recorded in selection.json (5.3).
Graduation: once the operator has confirmed $\ge 19$ production images of the class, it switches to its own conformal calibration set. Graduation images must never be evaluation images.
Operator corrections (stored as events): remove, rename, merge (pool embeddings, recompute mean with $n=10$), relabel an example, rollback to version $k$. Base classes are immutable in the UI.

Gate parameters [APPROVED-POLICY]:
- $\alpha_{\text{cons}} \in \{0.005, 0.01, 0.02, 0.05\}$
- $q_{\text{BH}} \in \{0.05, 0.10\}$
- $\epsilon_{\text{max}} \in \{0.02, 0.05, 0.10, 0.20\}$

---

## PART 5 — DATA SPLITS, REHEARSAL, SELECTION (Restored Text D.1 & D.2)

### 5.1 Partition per run (D.1)
Stratified, seeded, hash-disjoint pools; persisted as JSON manifests (paths + sha256 only; never copies of images).
- **Known class (300 images each)**:
  `train 120` | `early_stop 30` | `sel_cal 30` | `sel_eval 40` | `cal 40` | `test 40`
- **Held-out class (300 images each)**:
  `enroll_pool 50` | `unk_test 125` | `new_test 125`

**Allowed uses (strictly enforced)**:
- `train`: B1 weights, Gaussian means/covariance, mixture sub-means, cross-fitted weights, linear-probe baseline
- `early_stop`: B1 early stopping ONLY
- `sel_cal`: rehearsal calibration, rehearsal enrollment draws, injection sources
- `sel_eval`: rehearsal evaluation, gate tuning, injection experiments, V2 diagnostics, demo known-class images
- `cal`: final conformal scores ONLY
- `test`: final known-class metrics ONLY
- `enroll_pool`: enrollment draws; demo held-out images (never both operator example and query)
- `unk_test`: unknown-detection metrics ONLY
- `new_test`: new-class metrics + H1/H1-s tests ONLY

Pools must be HASH-disjoint. Before freezing manifests, run a near-duplicate check across pools (perceptual hash + embedding distance) and report findings.

### 5.2 Data-access guard
Each pipeline stage declares which pools it may read; reading any other pool raises a `DataAccessError`. Unit-tested and enforced in all pipeline scripts and loaders.

### 5.3 Rehearsal and selection inside each run (D.2)
Rehearsal is conducted inside EACH run, using ONLY that run's known classes and `train`/`early_stop`/`sel_cal`/`sel_eval`. Candidate space:
$$\text{backbone } \{B0, B1, B2\} \times \text{feature } \{L4, L234 \mid CLS, CLS\_patch\} \times \text{combination } (a)/(b) \times \text{mixture } k \in \{1, 2, 3\}.$$

For each candidate and each known class $k$ as pseudo-unknown:
1. **B1 only**: Fine-tune a FRESH backbone (fixed configuration) on `train` of the other known classes, early-stop on their `early_stop`. B0 and B2 require no training.
2. Fit Gaussians (+ mixture, + cross-fitted weights) on their `train`.
3. Calibrate on their `sel_cal`.
4. Score `sel_eval`: other classes = known, $k$ = unknown. Record AUROC ($\min_c \tilde{d}_c$), FRR at $\alpha=0.05$, ambiguity rate.
5. **Pseudo-enrollment**: 10 draws of 5 from $k$'s `sel_cal`; evaluate on $k$'s `sel_eval` (40 samples). Record for BOTH H1 and H1-s: recall, coverage/FRR vs $\alpha$, capture of other classes' `sel_eval`, $\hat{\epsilon}$ vs observed confusion.
6. **Gate tuning**: Evaluate on clean / INJECTED OUTLIER (replace 1 of 5 with another known class) / KNOWN-LOOKALIKE (replace 2 of 5 with the same other class) pseudo-enrollments, rehearsal pools only.

**Pre-registered Selection Rule**:
- **Primary criterion**: Mean rehearsal AUROC ($\min_c \tilde{d}_c$).
- **Tie-break within 0.01**: Mean pseudo-enrollment recall.
- **Exclusion filter**: Discard configurations with rehearsal known FRR $> 0.08$ at $\alpha = 0.05$.
- **H1 vs H1-s**: Choose the variant with smaller mean $|\text{rehearsal FRR} - \alpha|$ averaged over $\alpha \in \{0.05, 0.10\}$; ties go to H1.
- **Independence**: Selection is per run. NEVER pool selection across runs.

**Gate Parameter Rules**:
- $\alpha_{\text{cons}} \in \{0.005, 0.01, 0.02, 0.05\}$ maximising INJECTED OUTLIER detection subject to false-flag $\le 0.10$ on clean sets.
- $q_{\text{BH}} \in \{0.05, 0.10\}$ likewise for KNOWN-LOOKALIKE.
- $\epsilon_{\text{max}} \in \{0.02, 0.05, 0.10, 0.20\} =$ smallest value for which rehearsal enrollments with $\hat{\epsilon} \le \epsilon_{\text{max}}$ had observed confusion $\le 0.10$. If unmet, choose lowest false-flag value and record "constraint unmet".

---

## PART 6 — EXPERIMENTS, BASELINES, VERIFICATION (Restored Text D.3–D.6)

### 6.1 Experiments (D.3)
- **E1**: $H = \{Sc\} \times \text{seeds } 0\text{--}4$ (5 runs).
- **E2**: Each of the 6 classes $\times \text{seeds } 0\text{--}4$ (30 runs, including E1).
- **E3**: As E2, enrollment drawn ONLY from `enroll_pool` images that the run's frozen artifact rejected (empty set); if $< 5$ available -> record an `E3_SHORTFALL` row with the count, never pad.
- **E4**: All 30 ordered pairs $(h_1, h_2) \times \text{seeds } 0\text{--}4$ (150 runs), 4 known classes, own rehearsal; enroll $h_1$; measure $h_2$ rejection before/after, $h_2$ absorption into $h_1$, $h_1$ recall.

Every draw is enrolled and evaluated regardless of gate outcome; gate outcome is recorded; gate-conditional analyses are SECONDARY. Technical failures are logged, never silently resampled. Any compute-forced subset must be random, seeded, pre-declared, and disclosed.

### 6.2 Baselines
- Closed-set argmin
- Maximum Softmax Probability (MSP)
- Max-logit
- Energy score (B1 head, or linear probe $C=1.0$ on standardised train features for B0/B2)
- Cosine centroid
- k-NN ($k=5$)
- Mahalanobis without conformal
- NovaSight (inflated Mahalanobis + Mondrian conformal)

### 6.3 Reporting Tables (D.4)
- **T1**: Closed-set accuracy, per-class Precision/Recall/F1.
- **T2**: AUROC, AUPR, FPR@95%TPR with 1,000-resample image bootstrap CIs, all baselines (each with its native score).
- **T3**: At $\alpha = 0.05$: `empty_rate`, `ambiguous_rate`, `known_FRR`, `unk_rejection`, `operator_routing_rate`; confusion matrix with separate `"ambiguous"` and `"unknown"` columns.
- **T4**: Empirical vs nominal coverage over $\{0.01, 0.02, 0.05, 0.10, 0.20\}$ plus the Beta spread.
- **T5**: Per draw and mean $\pm$ sd: `new_class_coverage`, `new_class_recall`, `new_class_FRR` (H1 and H1-s), known test accuracy before/after (paired McNemar per run, descriptive), known->new capture, ambiguity change.
- **T5b**: Secondary gate tables.
- **T6**: LOCO summaries.
- **T7**: E3 (including shortfalls) and E4 pairs.
- **Curves**: OSCR curve, Risk–Coverage curve.

### 6.4 Mathematical Hypotheses (Pre-registered)
- **H1 & H1-s**: For $\alpha \in \{0.05, 0.10\}, \delta = 0.03$, per held-out class estimate the mean new-class FRR on `new_test` with a hierarchical bootstrap over (seed, draw, image):
  - *Supported on this dataset*: Macro point estimate in $[\alpha-\delta, \alpha+\delta]$ and no class interval entirely outside.
  - *Class-dependent*: Macro estimate in the band but $\ge 1$ class interval entirely outside (name classes; show V2 diagnostics).
  - *Falsified on this dataset*: Macro estimate outside the band and $\ge$ half of classes have intervals entirely outside.
  - Report per-draw Clopper–Pearson intervals and the share of draws whose interval excludes the band.
- **H2**: Spearman $\rho$ and calibration slope between $\hat{\epsilon}$ and observed confusion, pooled over all E2 draws, with bootstrap intervals. Fails if the $\rho$ interval includes 0 or the slope interval excludes 1.

### 6.5 Verification Stages (D.5)
- **V1 synthetic**: Proposition 1, Proposition 2 (draw shift $\approx \text{sd}(Q)/6$, variance ratio $\approx 0.97$), inheritance FRR, $\Delta^2$ unbiasedness, heteroscedastic $\{0.5, 0.8, 1.25, 2\}$, skewed/non-negative, bimodal new class.
- **V2 on sel data only**: Per-class score-scale ratios with bootstrap intervals, effective dimension, QQ plots.
- **V3 rehearsal**: Verification of candidate ranking.
- **V4 final test**: Confirmatory evidence.

### 6.6 Reporting Rules (D.6)
Effect sizes. Per-run image-bootstrap CIs; per-class seed values with mean and range; macro means; paired differences; direction counts. Corrected resampled t-test only in an appendix, within class, labelled approximate. No Wilcoxon across runs. Holm adjustment wherever multiple p-values appear. Never use the phrase "statistically significant".

---

## PART 7 — DEPLOYMENT & VERIFICATION (Restored Text D.7)

Primary run pre-declared $H=Sc$, split seed 0, init seed 0:
$$\text{selection} \rightarrow \text{backbone} \rightarrow (\text{cross-fit}) \rightarrow \text{Gaussian (train)} \rightarrow \text{conformal (cal)} \rightarrow \text{freeze + provenance} \rightarrow \text{evaluate (read-only)} \rightarrow \text{deploy unchanged.}$$
No refitting after evaluation. Never switch seed or configuration based on test results.
`scripts/verify_artifact.py`: Recomputes pool hashes; asserts no `test`, `unk_test`, or `new_test` hash exists among fitting/calibration inputs; checks `cal` was used only in calibration; rebuilds and verifies bit-identical hashes; confirms no test-pool image exists in the app bundle.

---

## PART 8 — INTERACTIVE APPLICATION (Restored Text D.8)

Tabs:
1. `Inspect`
2. `Operator queue`
3. `Enroll`
4. `Memory & audit`
5. `Evidence`

In the APPLICATION:
- `REJECT` strictly blocks enrollment.
- `REVIEW` requires a written operator acknowledgement note.
- $\alpha$ is fixed to the artifact's pre-declared value (slider only if labelled exploratory).
- Session sandbox with export/import capability.
- Demo images drawn ONLY from `sel_eval` and the primary run's `enroll_pool`.
- Display shows prediction set, p-value bars, nearest examples labelled "not a cause", the G1 reliability line, and the inherited-calibration caveat.
- Graduation occurs at $\ge 19$ confirmed production images (never evaluation images).

---

## PART 9 — CLARIFICATIONS & APPROVED DECISIONS (D1–D11)

*Approved by project owner on 22 October 2026. All D4–D7 changes are post-hoc relative to Sc/s0 and must be committed before any Gate 6 run reads its test pools. Where Part 9 conflicts with Parts 4–8, Part 9 governs.*

- **D1 (v1.0 Status)**: The v1.0 $Sc / s=0$ run is formally classified as `HISTORICAL_NONCOMPLIANT` and excluded from all confirmatory claims due to test pool leakage (`unk_test[:40]`) and duplicate pool straddling.
- **D2 (Splits v2)**: Splits v2 must be duplicate-group-aware (treating `Pa_101.bmp` and `Pa_105.bmp` as an atomic unit assigned to the same pool) and strictly hash-disjoint, persisted to `data/manifests/v2/`. v1 manifests remain untouched for auditability.
- **D3 (Significance Level)**: $\alpha = 0.05$ is fixed for all operational decisions; the $\alpha$ grid $\{0.01, 0.02, 0.05, 0.10, 0.20\}$ is evaluated only for Table T4 coverage diagnostics. $\alpha$ is never a candidate selection dimension.
- **D4 (Decision Rule & Exploratory Metric)**: Retain the specification decision rule (empty -> unknown, singleton -> class, 2+ -> ambiguous). Additionally report, as an EXPLORATORY metric, an "accept $\hat{c}$ when the set is ambiguous and contains $\hat{c}$" summary, computed only by code.
- **D5 (Gate Novelty Check N)**: Novelty check $N$ uses pooled-calibration p-values against $S_{\text{pool}}$ (resolution floor $1/201$).
- **D6 (Gate Validity Check V)**: Remove the $\text{cosine} \ge 0.95$ embedding near-duplicate rule; enforce exact SHA-256 and perceptual-hash (imagehash) checks.
- **D7 (Gate Overlap Check O)**: Implement the parametric bootstrap for the $\hat{\epsilon}$ range under the Gaussian model as specified.
- **D8 (Compute Design)**: D8 = pre-declared random subset on CPU; subset seed 20261022, drawn and committed before any Gate 6 run.
- **D9 (Backbone B2)**: D9 = keep B2 NOT RUN; may be revisited only via a pinned hub commit and weight sha256 with verified licence, as a separate approved change.
- **D10 (Hackathon Submission Scope)**: Submit the current application with an honest status section; protocol v1.1 work continues after submission and makes no claims in the submission beyond what committed files support.
- **D11 (Bootstrap Confidence Intervals)**: Remove unsupported bootstrap CIs from `README.md`; re-generate all confidence intervals strictly from committed code outputs at Gate 5/6.
- **D8 (amended 22 Oct, before any Gate 4–6 run)**: experiments run on Google Colab GPU, full design. If the full matrix cannot complete, the remaining runs are a random subset drawn with seed 20261022 and committed before their execution. Device (CPU/GPU, CUDA/cuDNN versions) is recorded in every provenance file. GPU results are not expected to be bit-identical to CPU.
