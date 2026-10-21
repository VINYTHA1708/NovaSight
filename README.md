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

| Metric | Value |
|--------|-------|
| T1 Closed-set accuracy | 1.0000 |
| T2 AUROC (min_c dt_c) | 0.9829 |
| T3 Known FRR (α=0.05) | 0.0150 |
| T3 Unk rejection rate | 0.8880 |
| T5 Mean recall H1 | 0.1296 |

> B0 is the frozen pretrained baseline. B1 (fine-tuned) and B2 (DINOv2) will be selected by the rehearsal protocol.

## Tests

```bash
python -m pytest tests/ -v          # 48 tests, all pass
```

## Project Structure

```
NovaSight/
├── app.py                    # Gradio UI (M3)
├── configs/default.yaml      # All parameters + approved_policy section
├── docs/RESEARCH_SPEC.md     # Verbatim research specification (Parts 4-9)
├── DEV_LOG.md                # Timestamped experiment log
├── scripts/
│   ├── build_index.py        # Index NEU-CLS, run sanity scan
│   ├── make_splits.py        # Generate 30 pool manifests
│   ├── cache_features.py     # Extract and cache backbone features
│   ├── build_artifact.py     # Fit Gaussian + calibrate conformal + freeze
│   └── evaluate.py           # Evaluate frozen artifact on test/enroll pools
├── src/
│   ├── config.py             # Config loader
│   ├── data/                 # index, splits, guard, transforms
│   ├── models/               # backbones, finetune, gaussian, conformal,
│   │                         # baselines, gate, memory
│   ├── eval/                 # (metrics, confusion matrix -- coming S6)
│   └── explain/              # Nearest-neighbour + score breakdown
├── data/manifests/           # 30 pre-committed pool manifests (H=x_s=y.json)
├── artifacts/base/           # Frozen base artifact (git-ignored weights)
├── reports/
│   ├── sanity_scan.json      # Application sanity scan (all 1800 images)
│   ├── run_registry.csv      # 210 planned experiment runs
│   └── results_*.json        # Evaluation results per run
└── tests/
    ├── test_s1.py            # 20 tests: index, splits, guard, sanity scan
    ├── test_s2.py            # 16 tests: preprocessing, backbones, Gaussian, conformal
    └── test_s4.py            # 12 tests: memory, enrollment, gate
```
