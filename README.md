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

## Primary Run Results (B0/L4, Sc held-out, seed 0)

| Metric | Value | 95% Bootstrap CI |
|--------|-------|------------------|
| T1 Closed-set accuracy | **1.0000** | [1.0000, 1.0000] |
| T2 AUROC (min_c dt_c) | **0.9829** | [0.9680, 0.9950] |
| T2 FPR@95%TPR | **0.0750** | — |
| T3 Known FRR (α=0.05) | **0.0150** | — |
| T3 Unk rejection rate (α=0.05) | **0.8880** | — |
| T5 Mean recall H1 (10 draws) | **0.1296** | — |

### Confusion Matrix with Unknown Column (M2 Requirement)

Evaluated on known test set (200 images, 40 per class) + held-out unknown test set (125 images) at α=0.05:

| True \ Pred | crazing | inclusion | patches | pitted_surface | rolled-in_scale | **unknown** |
|---|---|---|---|---|---|---|
| **crazing** | 0 | 0 | 0 | 0 | 0 | 40 |
| **inclusion** | 0 | 0 | 1 | 0 | 0 | 39 |
| **patches** | 0 | 0 | 23 | 0 | 0 | 17 |
| **pitted_surface** | 0 | 0 | 1 | 0 | 0 | 39 |
| **rolled-in_scale** | 0 | 0 | 0 | 0 | 0 | 40 |
| **unknown** (scratches) | 0 | 0 | 6 | 0 | 0 | **119** |

*95.2% of true unknown defect samples (119/125) are successfully identified as unknown and routed for operator review.*

### Model Selection Protocol (`scripts/select_model.py`)

- **Candidates evaluated**: B0 (frozen ImageNet ResNet-18) vs B1 (domain fine-tuned ResNet-18) across L4 features, mixture components $k \in \{1, 2, 3\}$, and significance levels $\alpha \in \{0.01, 0.02, 0.05, 0.10, 0.20\}$.
- **Validation data only**: Selected strictly on `sel_cal` and `sel_eval` pools. **Zero test data used.**
- **Selected configuration**: **B0/L4, k=1, α=0.01** (composite = 0.0340, AUROC = 0.9701).

## Tests

```bash
python -m pytest tests/ -v          # 62 tests, all pass (100%)
```

- `tests/test_s1.py`: 20 tests (index, stratified splits, pool disjointness, split access guard, sanity scan)
- `tests/test_s2.py`: 16 tests (transforms, ResNet backbones, Ledoit-Wolf Gaussian, conformal calibration)
- `tests/test_s4.py`: 12 tests (memory management, 3-gate 5-shot enrollment, versioning, event log replay)
- `tests/test_s6.py`: 14 tests (confusion matrix with unknown column, AUROC/FPR@95, FRR/FDR, bootstrap CIs)

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

