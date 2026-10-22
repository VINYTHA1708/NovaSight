---
title: NovaSight
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
python_version: "3.12"
---

# NovaSight — Open-Set Steel Defect Inspector

**NexAI Hack 2026 (British Council) · Problem PS 2.3 "Unknown-Defect Steel Inspector" · Track 2 Manufacturing QA**

NovaSight is a research-grade open-set steel-defect inspector trained on the [NEU-CLS](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) dataset.

## Mandatory Requirements (M1/M2/M3)

| ID | Requirement | Status |
|----|-------------|--------|
| **M1** | Classifier trained on 5 of 6 NEU-CLS classes; held-out class stated in README | ✅ **Held-out class: Scratches (Sc)** |
| **M2** | Unknown-detection method flags held-out images as "unknown"; shown in confusion matrix | ✅ Mondrian conformal + min inflated Mahalanobis |
| **M3** | UI: upload image → defect class or "unknown, send to operator"; operator labels 5 examples; system adds class without full retraining | ✅ Gradio app (`app.py`) |

## System Architecture

```
NEU-CLS (1800 BMP, 300/class)
    ↓ stratified seeded splits (S1)
 train │ early_stop │ sel_cal │ sel_eval │ cal │ test   ← per known class
 enroll_pool │ unk_test │ new_test                       ← held-out class
    ↓ B0/B1/B2 feature extraction + cache (S2)
 GaussianModel (Ledoit-Wolf shrinkage, mixture k∈{1,2,3}) (S3)
    ↓ calibration on CAL
 ConformalScores (Mondrian, class-conditional) (S3)
    ↓ frozen artifact (artifacts/base/)
 NovaSightMemory + EventLog (S4) ← enrollment without retraining
    ↓
 Gradio UI (app.py) ← Inspector / Operator / Memory / Provenance
```

## Approved Policies (pre-registered, fixed before any experiment)

Stored in `configs/default.yaml` under `approved_policy:` and recorded in `DEV_LOG.md` as *"approved before any experiment; not tuned on test data"*.

| Policy | Value |
|--------|-------|
| B1 fine-tune | layer4 unfrozen; backbone_lr=1e-4, head_lr=1e-3; AdamW wd=1e-4; batch=32; max_epochs=15; patience=3 |
| Augmentation | RandomHorizontalFlip, RandomVerticalFlip, ColorJitter(brightness=0.1, contrast=0.1) |
| Novelty score | min_c dt_c(x) (min inflated Mahalanobis over known classes) |
| App sanity checks | min_side=32px, near_constant_std=1.0; applied to UI input ONLY |
| H1 vs H1-s tiebreak | H1 |
| Evaluation enrollment | all 10 draws enrolled regardless of gate outcome |
| Linear probe | C=1.0, solver=lbfgs, multinomial, seed=42, train-only |

## Quickstart

```bash
# 1. Set up environment
python -m venv .venv
.venv\Scripts\activate   # Windows; or source .venv/bin/activate
pip install -r requirements.txt

# 2. Index dataset and generate splits (1800 images required in data/raw/)
python scripts/build_index.py
python scripts/make_splits.py

# 3. Extract features and build artifact (B0 backbone, primary run: Sc held-out, seed 0)
python scripts/cache_features.py --backbone B0 --feature L4 --held-out Sc --seed 0
python scripts/build_artifact.py --backbone B0 --feature L4 --held-out Sc --seed 0

# 4. Evaluate primary run
python scripts/evaluate.py --backbone B0 --feature L4 --held-out Sc --seed 0

# 5. Launch Gradio UI
python app.py
```

## Dataset Note

NEU-CLS contains one pair of content-identical images: `Pa_101.bmp` ≡ `Pa_105.bmp` (same sha256). This is a dataset property, not a code bug. Both files are kept in their evaluation pools. Reported in `reports/sanity_scan.json` and `DEV_LOG.md`.

## Primary Run Results (B0/L4, Sc held-out, seed 0) — Protocol v1.0

> **Status**: `HISTORICAL_NONCOMPLIANT` (Protocol v1.0). Excluded from confirmatory claims due to test pool leakage in candidate ranking (`unk_test[:40]`) and `Pa_101`/`Pa_105` duplicate straddling in v1 manifests. Results are exploratory only. Protocol v1.1 will re-evaluate on clean Splits v2 without test-pool leakage.

The historical deployment model was evaluated on the v1 test pool (200 known test images, 40 per class) and held-out unknown pool (125 scratches images) at significance level $\alpha = 0.05$:

| Metric | Value | Numerator / Denominator | 95% Bootstrap CI | Definition / Notes |
|--------|-------|-------------------------|------------------|--------------------|
| **T1 Closed-set accuracy** | **1.0000** | 200 / 200 | *(to be regenerated from code at Gate 5/6)* | Closed-set label $\hat{c} = \arg\min_c \tilde{d}_c(x)$ |
| **T2 AUROC (min_c dt_c)** | **0.9829** | — | *(to be regenerated from code at Gate 5/6)* | Ranking score $N(x) = \min_c \tilde{d}_c(x)$ |
| **T2 FPR@95%TPR** | **0.0750** | — | — | False positive rate at 95% true positive rate |
| **T3 Known Empty-Set Rejection (FRR)** | **0.0150** | 3 / 200 | — | Strict rejection: $\Gamma^\alpha(x) = \emptyset$ |
| **T3 Unknown Empty-Set Rejection** | **0.8880** | 111 / 125 | — | Strict unknown rejection: $\Gamma^\alpha(x) = \emptyset$ |
| **T3 Known Operator-Routing Rate** | **0.8750** | 175 / 200 | — | Non-singleton: $\Gamma^\alpha(x) \ne \{c\}$ (3 empty + 172 ambiguous) |
| **T3 Unknown Operator-Routing Rate** | **0.9520** | 119 / 125 | — | Non-singleton: $\Gamma^\alpha(x) \ne \{c\}$ (111 empty + 8 ambiguous) |
| **New-class coverage, H1 (10 draws) = 0.1296 (implies new-class FRR ≈ 0.87 at alpha=0.05; one run)** | 0.1296 | — | — | Evaluated on 10 random 5-shot enrollment draws on `new_test` |

### Confusion Matrix with Unknown Column (M2 Requirement)

Evaluated on known test set (200 images, 40 per class) + held-out unknown test set (125 images) at $\alpha=0.05$:

| True \ Pred | crazing | inclusion | patches | pitted_surface | rolled-in_scale | **unknown** |
|---|---|---|---|---|---|---|
| **crazing** | 0 | 0 | 0 | 0 | 0 | 40 |
| **inclusion** | 0 | 0 | 1 | 0 | 0 | 39 |
| **patches** | 0 | 0 | 23 | 0 | 0 | 17 |
| **pitted_surface** | 0 | 0 | 1 | 0 | 0 | 39 |
| **rolled-in_scale** | 0 | 0 | 0 | 0 | 0 | 40 |
| **unknown** (scratches) | 0 | 0 | 6 | 0 | 0 | **119** |

*Operational Definition in Confusion Matrix*:
- In the confusion matrix, "unknown" encompasses all samples requiring operator routing (both empty prediction sets $\Gamma^\alpha(x) = \emptyset$ and ambiguous sets $|\Gamma^\alpha(x)| \ge 2$), as neither can be automatically dispatched as a single known defect.
- **Known classes**: 25/200 samples are singletons ($\Gamma = \{c\}$) accepted automatically; 175/200 (87.5%) are routed to the operator as non-singletons (3 empty, 172 ambiguous).
- **Unknown class (scratches)**: 119/125 (95.2%) are routed to the operator (111 empty set, 8 ambiguous); only 6/125 are false acceptances into 'patches'.

### Model Selection Protocol & Protocol Limitation (`scripts/select_model.py`)

- **Candidates evaluated**: B0 (frozen ImageNet ResNet-18) vs B1 (domain fine-tuned ResNet-18) across L4 features, mixture components $k \in \{1, 2, 3\}$, and significance levels $\alpha \in \{0.01, 0.02, 0.05, 0.10, 0.20\}$.
- **Selected configuration**: **B0/L4, k=1, α=0.01** (composite = 0.0340, AUROC = 0.9701).
- **Selection Protocol Deviations (see `docs/PROTOCOL_DEVIATIONS.md`)**:
  - The post-implementation audit (`reports/POST_IMPLEMENTATION_AUDIT.md`) identified that `scripts/select_model.py` sub-sampled 40 images from `unk_test[:40]` as an unknown evaluation proxy during candidate ranking, despite `unk_test` being reserved strictly for test evaluation under Section 5.1 of `docs/RESEARCH_SPEC.md`.
  - The composite score formula $\text{AUROC} \times (1 - \text{FRR})$ and varying $\alpha$ during selection were ad-hoc Protocol v1.0 implementations, not pre-registered in Part 5.3. Under Protocol v1.1 (Decision D3), $\alpha = 0.05$ is fixed for all operational decisions.
  - Consequently, the model selection cannot be claimed as fully compliant with the specification's clean split boundaries.
  - The evaluated primary artifact is preserved as-is for audit transparency, and no test pool data will be used for any future tuning, calibration, or selection.

### Backbone B2 (DINOv2) Status: NOT RUN

- **Status**: `NOT RUN`.
- **Model**: DINOv2 ViT-S/14 (`dinov2_vits14`), Hub repo: `facebookresearch/dinov2`.
- **License**: Apache 2.0.
- **Details**: DINOv2 weights were not downloaded to local cache `data/processed/dinov2_cache/` due to offline environment constraints. Per Part 4.2 of `docs/RESEARCH_SPEC.md`, B2 is explicitly declared as "not run", and no empirical performance claims are made for B2.

## Experiment Matrix Scope & Status (`reports/run_registry.csv`)

| Experiment | Description | Matrix Size | Executed | Status |
|---|---|---|---|---|
| **E1** | Primary Run (held-out Scratches, seeds 0–4) | 5 runs | 1 run (seed 0) | Seed 0: HISTORICAL_NONCOMPLIANT; seeds 1–4 PLANNED |
| **E2** | Leave-One-Class-Out (6 classes $\times$ seeds 0–4) | 25 runs | 0 runs | PLANNED |
| **E3** | Queue-Sourced Enrollment | 30 runs | 0 runs | PLANNED |
| **E4** | Two-Unknown Ordered Pairs (30 pairs $\times$ seeds 0–4) | 150 runs | 0 runs | PLANNED |
| **Total** | | **210 runs** | **1 run** | **1 historical exploratory run (0.48%)** |

- Primary run: E1, held-out `Sc`, split seed 0 (Result SHA-256: `e2b3ef73b1e043595dae6355dfab13abb16cc5fa74d3d06fbb0594ed599f1361`).
- All remaining 209 runs are recorded as `PLANNED` in `reports/run_registry.csv`. No results have been fabricated.

## Tests

```bash
python -m pytest tests/ -v          # 62 tests, all pass (100%)
```

- `tests/test_s1.py`: 20 tests (index, stratified splits, pool disjointness, split access guard, sanity scan)
- `tests/test_s2.py`: 16 tests (transforms, ResNet backbones, Ledoit-Wolf Gaussian, conformal calibration)
- `tests/test_s4.py`: 12 tests (memory management, 3-gate 5-shot enrollment, versioning, event log replay)
- `tests/test_s6.py`: 14 tests (confusion matrix with unknown column, AUROC/FPR@95, FRR/FDR, bootstrap CIs)

> **Audit Note**: Passing 62 unit and integration tests confirms code functionality, data invariants, and mathematical formulas; it does not substitute for executing the 209 unexecuted benchmark experiment runs.

## Project Structure

```
NovaSight/
├── app.py                         # 4-tab Gradio UI (Inspector, Operator, Memory, Provenance) (M3)
├── configs/default.yaml           # All parameters + approved_policy section
├── docs/RESEARCH_SPEC.md          # Verbatim research specification (Parts 4-9)
├── DEV_LOG.md                     # Timestamped experiment log and approved policies
├── scripts/
│   ├── build_index.py             # Index NEU-CLS, run sanity scan
│   ├── make_splits.py             # Generate 30 pool manifests
│   ├── cache_features.py          # Extract and cache backbone features
│   ├── run_finetune.py            # B1 domain fine-tuning with early stopping
│   ├── select_model.py            # Validation-only model selection protocol
│   ├── build_artifact.py          # Fit Gaussian + calibrate conformal + freeze artifact
│   ├── evaluate.py                # Evaluate frozen artifact on test/enroll pools
│   └── gen_confusion_matrix.py    # Generate M2 confusion matrix with unknown column
├── src/
│   ├── config.py                  # YAML config loader with environment overrides
│   ├── data/                      # Indexing, stratified splits, access guard, transforms
│   ├── models/                    # Backbones (B0/B1), Gaussian, conformal, baselines, gate, memory
│   ├── eval/                      # Metrics, confusion matrix, bootstrap CIs
│   └── explain/                   # Nearest-neighbour and score breakdown
├── data/manifests/                # 30 pre-committed pool manifests (H=x_s=y.json)
├── artifacts/base/                # Frozen base artifact (gaussian, class_means, conformal, metadata)
├── reports/
│   ├── sanity_scan.json           # Application sanity scan (all 1800 images)
│   ├── selection_result.json      # B0 vs B1 candidate selection report
│   ├── confusion_matrix_*.json    # Confusion matrix with unknown column (M2)
│   └── results_*.json             # Full evaluation results
└── tests/
    ├── test_s1.py                 # 20 tests: data, splits, guard, scan
    ├── test_s2.py                 # 16 tests: representation, Gaussian, conformal
    ├── test_s4.py                 # 12 tests: memory, 3-gate enrollment, replay
    └── test_s6.py                 # 14 tests: metrics, confusion matrix, CIs
```

