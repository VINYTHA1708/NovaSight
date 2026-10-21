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

