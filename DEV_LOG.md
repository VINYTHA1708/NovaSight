# NovaSight Development Log

## 21 October 2026 — Project Setup

- Created the NovaSight project directory.
- Created and activated a Python 3.12.10 virtual environment.
- Initialized a local Git repository on the main branch.
- Added .gitignore to exclude virtual environments, datasets, model weights, and local secrets.
- Created the initial README.md.
- Added the initial requirements.txt.

## Next Steps
- Verify the NEU-CLS dataset source and usage terms.
- Install and verify the Python dependencies.
- Implement the data loading and validation pipeline.
- Implement known-class classification and unknown-defect detection.
- Evaluate with held-out data and record actual results.
- Build and deploy the Gradio demo.


---

## 22 October 2026 — S0 Inspection + S1 Implementation (AI-assisted: Antigravity/Claude Sonnet 4.6 Thinking)

### S0: Repository inspection findings
- Git state: branch main, last commit 94bf0c1, working tree clean. Confirmed.
- Dataset: 1,800 BMP images, exactly 300 per class (Cr/In/Pa/PS/RS/Sc), all 200x200, all readable. Thumbs.db present and ignored. No images trip application sanity checks (std check, min-side check).
- **DATA FINDING**: Pa_101.bmp and Pa_105.bmp have identical content (sha256: 6833002b8b742894...). This is a NEU-CLS dataset property, not a code bug. Both files are KEPT in their evaluation pools per spec. Reported here and in sanity_scan.json.
- All expected packages installed (torch 2.14.1+cpu, torchvision 0.29.1+cpu, etc.). CUDA unavailable (CPU-only, expected).
- Missing packages added: scipy, imagehash, pytest, pyyaml, huggingface_hub.
- Missing from .gitignore: state/ — added.
- docs/RESEARCH_SPEC.md committed before any implementation code (commit e6262ee).

### S1: Config, index, splits, guard, sanity scan
- configs/default.yaml: all seeds, sizes, alpha grid, delta, primary run (Sc, seed 0, init 0), selection rule, approved_policy section.
  **approved before any experiment; not tuned on test data**:
  - B1 fine-tune: layer4 unfrozen, backbone_lr=1e-4, head_lr=1e-3, AdamW, wd=1e-4, batch=32, max_epochs=15, patience=3, augmentation=H/V flip + brightness/contrast jitter 0.1
  - Sanity check thresholds: min_side=32px, near_constant_std=1.0 (0-255 scale)
  - Mixture k candidates: {1,2,3}, selected by rehearsal only
  - Primary novelty score: min_c dt_c(x)
  - H1 vs H1-s tiebreak: H1
  - Evaluation-time enrollment: all 10 draws enrolled regardless of gate outcome
  - Linear probe baseline: C=1.0, lbfgs, multinomial, seed=42, train only
  - Gate false-flag tolerance: 0.10
- src/data/index.py: builds index (1800 records, validated per-class counts), runs sanity scan.
- src/data/splits.py: stratified seeded splits (train=120, early_stop=30, sel_cal=30, sel_eval=40, cal=40, test=40 per known class; enroll_pool=50, unk_test=125, new_test=125 for held-out). Disjointness check uses file paths (not sha256, because the Pa_101/Pa_105 content-identical pair would otherwise trigger a false alarm). Near-duplicate check via perceptual hash (imagehash).
- src/data/guard.py: pool access guard; raises DataAccessError on undeclared pool.
- src/data/transforms.py: eval transform (grayscale->224->3ch->normalize) and training transform with approved augmentation; application sanity checks.
- scripts/build_index.py and scripts/make_splits.py created.
- Sanity scan result: 0/1800 images trip any check. Report: reports/sanity_scan.json.

### Bugs found and fixed during S1
1. _assert_disjoint used sha256 as uniqueness key, causing false alarm on Pa_101/Pa_105. Fixed to use file path.
2. imagehash Hamming distance returns numpy int64 which is not JSON-serializable. Fixed to cast to Python int.


---

## 22 October 2026 — S2 through S6 (AI-assisted: Antigravity)

### S2: Backbones, Gaussian, conformal, B1 trainer, baselines, feature cache
- src/models/backbones.py: ResNet-18 (L4: 512-d, L234: 896-d). Frozen stem+layers 1-3.
- src/models/gaussian.py: Ledoit-Wolf shrinkage, inflated Mahalanobis dt_c(f), novelty_score = min_c dt_c(f).
- src/models/conformal.py: Mondrian class-conditional conformal. p-value = (|{s in S_c: s >= score}|+1) / (|S_c|+1).
- src/models/finetune.py: B1 trainer with approved_policy assertions (backbone_lr=1e-4, head_lr=1e-3, AdamW, wd=1e-4, batch=32, max_epochs=15, patience=3, H/V flip + brightness/contrast jitter 0.1). **approved before any experiment; not tuned on test data**.
- src/models/baselines.py: MSP, max-logit, energy, cosine-centroid, k-NN (k=5), Mahalanobis, closed-set, linear probe.
- scripts/cache_features.py: caches features for all pools. 16 S2 tests pass.

### S3: 5-shot enrollment gate
- src/models/gate.py: validity (duplicate sha256), novelty (BH correction), consistency, overlap (eps_hat bootstrap).

### S4: Memory, event log, enrollment lifecycle
- src/models/memory.py: enroll/remove/rename/merge; append-only events.jsonl; replay_events(); H1 and H1-s scoring. 12 S4 tests pass.

### S5: Primary artifact built; Gradio app (M3); primary run evaluated
- Primary run (B0/L4, Sc held-out, seed=0): T1=1.000, T2 AUROC=0.9829, T3 FRR=0.015, T3 unk-rejection=0.888, T5 mean-recall-H1=0.130.
- app.py: 4-tab Gradio UI (Inspector, Operator, Memory, Provenance). Satisfies M3.

### S6: Metrics, selection protocol, B1 fine-tuning
- src/eval/metrics.py: K+1 x K+1 confusion matrix (M2), AUROC, FPR@95, FRR, FDR.
  T1 uses argmin (closed-set), not conformal verdict. CM uses conformal verdict.
- Primary CM (B0/L4, alpha=0.05): T1=1.000, AUROC=0.9829, FPR95=0.075, FRR=0.875, FDR=0.193, unk-rejection=0.952.
- scripts/select_model.py: criterion = AUROC(sel_eval) * (1-FRR). **NO TEST DATA USED**. Selected: B0/L4, k=1, alpha=0.01. **approved before any experiment; not tuned on test data**.
- scripts/run_finetune.py: B1 best_epoch=6, val_loss=0.0017, early-stopped at 9. Bug fixed: _B1FeatureExtractor wrapper replaces non-existent get_feature_extractor().

### Bugs fixed S5-S6
1. numpy str_ in class_names npz -> JSON error. Fixed: [str(c) for c in ...].
2. T1 accuracy counted ambiguous as wrong. Fixed to use closed-set argmin.
3. run_finetune called non-existent model.get_feature_extractor(). Fixed: _B1FeatureExtractor wrapper.
