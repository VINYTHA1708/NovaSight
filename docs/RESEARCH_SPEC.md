# NovaSight Research Specification

> **NOTICE (PROTOCOL v1.1)**: This document records the historical Protocol v1.0 specification.
> This file is a condensed version, not a verbatim copy, of the final master prompt Parts 4–9.
> The authoritative, updated specification restoring master sections D.1–D.8 and incorporating approved decisions D1–D11 is located at [docs/RESEARCH_SPEC_v1.1.md](RESEARCH_SPEC_v1.1.md).
> See [docs/PROTOCOL_DEVIATIONS.md](PROTOCOL_DEVIATIONS.md) for the complete record of v1.0 deviations and dispositions.

> This file was committed as required by Part 3.3 of the master implementation prompt
> (NOVASIGHT — MASTER IMPLEMENTATION PROMPT FOR ANTIGRAVITY, FINAL).
> It contains Parts 4–9 verbatim. It is the authoritative specification.
> If any instruction later appears to conflict with it, stop and report the conflict.
> Committed before any implementation code was written.

---

## PART 4 — METHOD SPECIFICATION

### 4.1 Preprocessing
Load as grayscale (convert RGB uploads to L) -> resize to 224
(bilinear) -> repeat to 3 channels -> backbone-specific normalisation
(ImageNet mean/std for ResNet; DINOv2's own normalisation for B2).
Input sanity checks (heuristic, not calibrated) — APPLICATION INPUT
ONLY: undecodable file, min side < 32 px, near-constant image
(grayscale pixel std < 1.0 on the 0–255 scale) [APPROVED-POLICY] ->
reject with a specific message.
These checks are NEVER applied to evaluation data and never silently
drop images. In S1, scan all 1,800 NEU images with the same checks and
write reports/sanity_scan.json (count + filenames of any image that
would trip). If any image trips, report it in README and DEV_LOG; it
stays in its evaluation pool.

### 4.2 Backbone candidates (all must be implemented and compared; no winner may be assumed)
B0: torchvision ResNet-18, IMAGENET1K_V1 weights, frozen.
B1: ResNet-18 from the same weights, fine-tuned with a linear head
    (5-way for final models, 4-way in nested rehearsal) and
    cross-entropy on known-class TRAIN data only; early stopping on
    EARLY-STOP data only.
    [APPROVED-POLICY] single fixed configuration (no hyperparameter
    grid): only layer4 unfrozen (stem–layer3 frozen); backbone LR 1e-4;
    head LR 1e-3; AdamW, weight decay 1e-4; batch 32; max 15 epochs;
    patience 3 on early-stop cross-entropy loss (restore best epoch);
    augmentation = horizontal and vertical flips + brightness/contrast
    jitter of +-0.1; model-initialisation seed from config (head init and
    data order). Document in README that this is a fixed engineering
    choice, not an optimised configuration. The same configuration is
    used for every nested rehearsal fine-tune and every final fine-tune.
B2: DINOv2 ViT-S/14 (torch.hub facebookresearch/dinov2,
    "dinov2_vits14"), frozen. This needs network access on first run;
    cache the weights and record the hash. Verify and record its
    licence in README. If the weights cannot be obtained, report B2
    as "not run". NEVER substitute another model under the B2 name.
Every backbone: record the weight source, sha256 and init seed. Load
from the local cache when present; fail with a clear message, not a
silent fallback, when absent. Everything must run on CPU; use a GPU
only if one is present, and never require it.

### 4.3 Feature options (compared in rehearsal)
ResNet (B0/B1): L4 = layer4 global-average-pooled (D=512);
  L234 = concatenation of GAP(layer2) 128 + GAP(layer3) 256 +
  GAP(layer4) 512 = 896, each block z-scored with TRAIN statistics.
DINOv2 (B2): CLS token; CLS concatenated with mean patch token.
Combination options for multi-block features:
  (a) one Gaussian on the concatenated vector (default);
  (b) per-block Gaussian scores combined by logistic regression whose
      weights are learned by 5-fold CROSS-FITTING WITHIN TRAIN ONLY:
      fit per-block Gaussians on 4 folds, score the held-out fold
      (known positives); create pseudo-unknown negatives by leaving
      one class out within the out-of-fold scores; fit the logistic
      regression on these out-of-fold scores only. The weights must be
      a function of TRAIN alone (unit test: perturbing any other pool
      leaves the weights bit-identical).
Do NOT reduce dimensionality with top-variance PCA. Low-variance
directions matter for novelty (Rippel et al. 2020).

### 4.4 Class-conditional Gaussian model (shared covariance)
Working ASSUMPTION (never present it as a fact about NEU):
  f | y=c ~ N(mu_c, Sigma), Sigma shared.
Estimate mu_c = class mean of TRAIN features (n_c = 120 per known
class). Residuals R = f_i - mu_hat_{y_i}. Sigma_hat = Ledoit-Wolf
shrinkage estimate computed from R only (assume_centered on
residuals). A = inverse(Sigma_hat), computed stably; assert positive
definiteness; log the condition number and the shrinkage intensity.
Score: d_c(f) = (f - mu_c)^T A (f - mu_c).
Inflated score: dt_c(f) = d_c(f) / (1 + 1/n_c).
Closed-set label: c_hat = argmin_c dt_c(f).
Mixture variant [APPROVED-POLICY]: k-means (seeded) sub-means per known
class fitted on TRAIN only, k in {1,2,3} (k=1 is the plain Gaussian),
sharing the same A; class score = min over sub-means, each sub-mean
inflated by (1 + 1/n_sub). k is a candidate dimension selected ONLY by
the rehearsal rule in 5.3. Enrolled classes always use a single mean.

### 4.5 Conformal decision (class-conditional / Mondrian)
For each known class c: S_c = { dt_c(x_i) : x_i in CAL, y_i = c }
(40 scores per class).
p_c(x) = (1 + #{s in S_c : s >= dt_c(x)}) / (|S_c| + 1).
Prediction set Gamma_alpha(x) = { c : p_c(x) > alpha }.
Output rule:
  empty set   -> "unknown, send to operator"
  {c}         -> class c
  2+ classes  -> "ambiguous between ..." -> operator review (also show c_hat).
Primary novelty score for ALL AUROC / AUPR / FPR@95 / OSCR computation,
for both rehearsal selection and final reporting [APPROVED-POLICY]:
N(x) = min_c dt_c(x) (over all classes currently in memory; for
multi-block option (b), the cross-fitted combined score). Conformal
p-values are NOT used for AUROC (their 1/41 granularity causes ties);
they drive only the decision rule (prediction sets, unknown/ambiguous
outputs, FRR at alpha, coverage).
Default alpha = 0.05; full grid {0.01, 0.02, 0.05, 0.10, 0.20}.
INVARIANT (must be unit-tested): adding or removing an enrolled class
never changes any known class's p-values (bit-identical).
S_pool = union of all S_c (200 scores); tau_hat = mean(S_pool).

### 4.6 Guarantee statement
G1 (known classes): if (A1) a test image of class c and class c's CAL
images are exchangeable given Y=c, and (A2) the score function does not
depend on CAL or test images, then P(p_c(X) <= alpha | Y=c) <= alpha.
Hence P(rejected as unknown | Y=c) <= alpha, and P(wrong single label
| Y=c) <= alpha. G1 holds on average over calibration sets. For a
fixed CAL set, coverage ~ Beta(n+1-l, l) with l = floor((n+1)alpha);
at n=40, alpha=0.05 this is Beta(39,2) (mean approx 0.951, sd approx 0.033).
Report this spread.
NOT guaranteed: (N1) unknown-detection power; (N2) prediction-set
size; (N3) behaviour under distribution shift or non-random selection;
(N4) anything about enrolled classes (see 4.8).

### 4.7 Five-shot enrollment
Inputs: exactly 5 images + name. Embed with the deployed backbone.
Checks:
 V (validity): decodable, size OK, not near-constant, no duplicate
   sha256, no near-duplicate embeddings, name non-empty and unique
   -> REJECT on failure.
 N (novelty): p_c(f_i) for every example i and known class c;
   Benjamini-Hochberg across all 5xK tests at level q_BH. If >= 3 of 5
   examples are consistent with the SAME known class c -> REJECT
   ("these look like c; label them as c?"). If 1-2 examples are ->
   REVIEW, naming them.
 C (consistency): for each i, the distance from f_i to the mean of the
   other four, inflated by (1 + 1/4), converted to a p-value against
   S_pool. Any p < alpha_cons -> REVIEW ("example i is unlike the
   other four").
 O (overlap): delta_hat = mu_new - mu_c;
   Delta2_hat = max(0, delta_hat^T A delta_hat
                       - tau_hat * (1/5 + 1/n_c));
   eps_hat_c = Phi(-sqrt(Delta2_hat)/2), a plug-in approximation that
   treats Sigma_hat as Sigma. Report a parametric-bootstrap range under
   the Gaussian model. Any eps_hat_c > eps_max -> REVIEW ("estimated
   ~X% confusion with c; consider merge").
Outcomes: ACCEPT; REVIEW (operator must acknowledge each warning with
a written note, or replace the flagged examples); REJECT (cannot be
overridden). (In EVALUATION, the gate is recorded but never blocks
enrollment; see 6.1.)
On acceptance: mu_new = mean of the 5 embeddings; covariance = shared
A; n = 5, so inflation is 1.2.
 H1 (plain inheritance): p_new(x) uses S_pool with dt_new = d_new / 1.2.
 H1-s (scale-adaptive): r_hat = (mean of the 5 leave-one-out
   distances, each inflated by 1 + 1/4) / tau_hat;
   score = dt_new / r_hat, compared against S_pool.
 The deployed app uses the variant recorded in selection.json (5.3).
Graduation: once the operator has confirmed >= 19 production images
of the class, it switches to its own conformal calibration set.
Graduation images must never be evaluation images.
Operator corrections (all stored as events): remove, rename, merge
(pool the stored embeddings, recompute the mean with n = 10), relabel
an example, rollback to version k. Base classes are immutable in the UI.
Gate parameters [APPROVED-POLICY]:
 grids: alpha_cons in {0.005, 0.01, 0.02, 0.05};
        q_BH in {0.05, 0.10};
        eps_max in {0.02, 0.05, 0.10, 0.20}.
Injection definitions (rehearsal pools only, seeded):
 INJECTED OUTLIER: in a 5-image pseudo-enrollment set of class k,
   replace 1 image with an image of a different known class.
 KNOWN-LOOKALIKE: replace 2 images with images of the SAME other class.
 Clean set: 5 images of class k, no replacement.
The 0.10 tolerances are documented policy choices, not derived values.

### 4.8 Calibration inheritance — status of the mathematics
Fact 1 (exact under the Gaussian model): class means are independent
  of within-class residuals, so A (computed from residuals only) is
  independent of all class means and of new-class data.
Proposition 1 (exact under the model, averaged over mean-estimation
  error b = mu_hat - mu ~ N(0, Sigma/n)): for any fixed A,
  d/(1+1/n) has the same distribution as Q = w^T A w with
  w ~ N(0, Sigma). Unshrunk inverse covariance would be biased by
  (N-K)/(N-K-D-1) (~7.3 at D=512) or undefined at D=896.
Proposition 2 (exact under the model, conditional on one draw): the
  mean of the inflated score shifts by a random amount with
  sd(Q)/(n+1), about sd(Q)/6 for n=5, INDEPENDENT of D. Variance
  ratio is (2 + 4/n)/(2(1+1/n)^2) approx 0.97 at n=5.
Approximation (NOT exact): if Q is roughly Gaussian, a +-1 sd draw
  shift at alpha=0.05 gives per-draw FRR roughly 3.5%-7.0%.
Known failure modes: heteroscedastic classes; non-Gaussian features;
  unrepresentative enrollment sets; multimodal new classes.
There is NO distribution-free guarantee for enrolled classes. H1 and
H1-s are hypotheses, tested in 6.4.

---

## PART 5 — DATA SPLITS, REHEARSAL, SELECTION

### 5.1 Partition per run
Stratified, seeded, hash-disjoint pools; persisted as JSON manifests
(paths + sha256 only; never copies of images).
Known class, 300 images each:
  train 120 | early_stop 30 | sel_cal 30 | sel_eval 40 | cal 40 | test 40
Held-out class, 300 images each:
  enroll_pool 50 | unk_test 125 | new_test 125
Allowed uses:
  train      : B1 weights, Gaussian means/covariance, mixture sub-means,
               cross-fitted weights, linear-probe baseline
  early_stop : B1 early stopping ONLY
  sel_cal    : rehearsal calibration + rehearsal enrollment draws +
               injection sources
  sel_eval   : rehearsal evaluation, gate tuning, injection experiments,
               V2 diagnostics, demo known-class images
  cal        : final conformal scores ONLY
  test       : final known-class metrics ONLY
  enroll_pool: enrollment draws; demo held-out images
  unk_test   : unknown-detection metrics ONLY
  new_test   : enrolled-class recall + H1/H1-s tests ONLY
Before freezing manifests, run a near-duplicate check across pools
(perceptual hash + embedding distance) and report findings.

### 5.2 Data-access guard
Each pipeline stage declares which pools it may read; reading any other
pool raises an error. Unit-tested.

### 5.3 Rehearsal and selection
Candidate space: backbone {B0, B1, B2} x feature option (4.3) x
combination (a)/(b) x mixture k {1,2,3}.
PRE-REGISTERED SELECTION RULE:
  primary = mean rehearsal AUROC (min_c dt_c); tie-break within 0.01 =
  mean pseudo-enrollment recall; discard configs with rehearsal known
  FRR > alpha + 0.03 at alpha = 0.05.
  [APPROVED-POLICY] H1 vs H1-s: choose the variant with smaller mean
  |rehearsal FRR - alpha| averaged over alpha in {0.05, 0.10}; ties to H1.
Record every candidate's rehearsal metrics, the chosen configuration,
the chosen inheritance variant and the gate parameters in selection.json.
Selection is per run. NEVER pool selection across runs.

---

## PART 6 — EXPERIMENTS, BASELINES, VERIFICATION, REPORTING

### 6.1 Experiments
E1 primary: H = {Sc}, seeds 0-4.
E2 LOCO: H = each of the 6 classes, seeds 0-4 (30 runs, including E1).
E3 queue-sourced enrollment: drawn from enroll_pool images rejected by
   the artifact. E3_SHORTFALL if < 5 rejected images available.
E4 two-unknowns: all 30 ordered pairs x seeds 0-4.
EVALUATION-TIME ENROLLMENT RULE [APPROVED-POLICY]: Every one of the 10
draws per run is ENROLLED and evaluated regardless of gate outcome.

### 6.2 Baselines
closed-set; MSP; max-logit; energy; cosine centroid; kNN (k=5);
Mahalanobis without conformal; NovaSight.
Linear probe [APPROVED-POLICY]: multinomial logistic regression on
standardised frozen features, C=1.0, fixed solver/seed, train only.

### 6.3 Metrics / tables
T1: closed-set accuracy, per-class P/R/F1.
T2: AUROC, AUPR, FPR@95%TPR with bootstrap CIs (1,000 resamples).
T3: at alpha=0.05: rejection rate, known FRR, ambiguity, confusion matrix.
T4: empirical vs nominal coverage over alpha grid + Beta spread.
T5: post-enrollment (all draws): recall, FRR, McNemar, capture, ambiguity.
T5b (SECONDARY): gate outcome distribution; association with recall.
T6: LOCO summaries of T2/T5.
T7: E3 (including E3_SHORTFALL rows) and E4 results.

### 6.4 Hypotheses and decision rules (pre-registered)
H1/H1-s: delta = 0.03. Supported / Class-dependent / Falsified based on
  macro point estimate band and class intervals.
H2: Spearman rho and calibration slope between eps_hat and observed
  confusion. Fails if rho interval includes 0 or slope excludes 1.

### 6.5 Verification programme
V1 synthetic; V2 real-data diagnostics (SEL only); V3 rehearsal;
V4 final test. ONLY V4 counts as evidence.

### 6.6 Statistical reporting policy
Primary: effect sizes. Never write "statistically significant"; use
"indicative". Never omit failed runs. Secondary analyses labelled.

---

## PART 7 — DEPLOYMENT PROCEDURE

Primary run pre-declared: held-out = Sc, split seed = 0, model-init seed = 0.
Steps: rehearsal+selection -> backbone -> cross-fitted weights ->
Gaussian fit -> conformal calibration -> freeze artifacts/base/ ->
evaluate -> deploy unchanged.
NO refit after evaluation.

---

## PART 8 — APPLICATION (Gradio, CPU)

Five tabs: Inspect | Operator queue | Enroll | Memory & audit | Evidence.
Persistence: state/events.jsonl (append-only). Memory = deterministic
replay of event log over base artifact.
HF Spaces: ask user before any push.

---

## PART 9 — REPOSITORY LAYOUT

```
configs/default.yaml
src/config.py
src/data/{index,splits,transforms,guard}.py
src/models/{backbones,finetune,gaussian,conformal,gate,memory,baselines}.py
src/eval/{rehearsal,metrics,protocols,diagnostics,simulate}.py
src/explain/evidence.py
src/utils/{logging,hashing,io}.py
scripts/{build_index,make_splits,cache_features,rehearse,select_config,
         build_artifact,evaluate,report,verify_artifact}.py
app.py
artifacts/base/
state/           (git-ignored)
reports/         (JSON + PNG + run_registry.csv + sanity_scan.json)
docs/RESEARCH_SPEC.md
docs/ARCHITECTURE.md
tests/
```
