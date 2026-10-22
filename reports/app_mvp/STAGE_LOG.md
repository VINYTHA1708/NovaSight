# NovaSight Application Track — MVP Stage Log

## Stage S0: Inspection & Baseline
- **Date**: 2026-10-22
- **Initial Commit on main**: `5e970a8` (clean working tree)
- **Branch Created**: `app-mvp`
- **Python Environment**: `.venv\Scripts\python.exe` (Python 3.12.10)
- **Baseline Test Suite**:
  - Command: `.venv\Scripts\python.exe -m pytest -q`
  - Output: `64 passed, 1 warning in 86.00s`
- **Cached Feature Verification**:
  - `data/processed/feature_cache/B0_L4_H=Sc_s=0_Cr_sel_eval.npz`: EXISTS
  - `data/processed/feature_cache/B0_L4_H=Sc_s=0_In_sel_eval.npz`: EXISTS
  - `data/processed/feature_cache/B0_L4_H=Sc_s=0_Pa_sel_eval.npz`: EXISTS
  - `data/processed/feature_cache/B0_L4_H=Sc_s=0_PS_sel_eval.npz`: EXISTS
  - `data/processed/feature_cache/B0_L4_H=Sc_s=0_RS_sel_eval.npz`: EXISTS
  - `data/processed/feature_cache/B0_L4_H=Sc_s=0_Sc_enroll_pool.npz`: EXISTS
- **Status**: PASSED S0.

## Stage S1: Real-Weight ONNX Equivalence
- **Date**: 2026-10-22
- **ONNX Export**:
  - Script: `scripts/export_onnx.py`
  - Model: `models/onnx/resnet18_b0_l4.onnx` (44,696,360 bytes, opset 17)
  - ONNX SHA-256: `388c2afc2a63797524d93f0c89469771ad1c11e96e4b08bcb813e0a2a134ff4c`
  - State Dict SHA-256: `de620ffcedf6c2818d8b6575ad3b3fc89a41351a1588ef8c0d904defc28ebf8e`
  - Metadata: `models/onnx/MODEL_CARD.json`
- **Runtime Package**:
  - Implemented `src/runtime/` (`preprocess.py`, `onnx_extractor.py`, `artifact.py`, `scoring.py`).
  - Zero imports of `torch`, `torchvision`, or `sklearn` in runtime modules.
- **Equivalence Verification**:
  - Script: `scripts/check_onnx_equivalence.py` on 250 allowed images (200 `sel_eval` + 50 `held_enroll_pool` Sc)
  - (a) Preprocessing max abs diff: `0.00e+00` (Threshold <= 1e-6) -> PASS
  - (b) Live extractor diff: `abs=1.05e-05`, `rel=6.65e-02` (Threshold <= 1e-4) -> PASS
  - (c) Feature cache diff: `1.05e-05` (Threshold <= 1e-4) -> PASS
  - (d) Inflated score max rel diff: `0.00e+00` (Threshold <= 1e-6) -> PASS
  - (e) Conformal p-value mismatches: `0` (Threshold == 0) -> PASS
  - (f) Decision / verdict mismatches: `0` (Threshold == 0) -> PASS
  - (g) Median CPU latency: `781.25 ms/image` (1 thread)
  - Saved Report: `reports/app_mvp/onnx_equivalence.json`
- **Tests Added**:
  - `tests/test_runtime.py`: clean subprocess imports check, synthetic scoring match, forced-mismatch exit check -> `3 passed`.
  - Full test suite: `67 passed, 1 warning in 111.93s`.
- **Status**: PASSED S1.

## Stage S2: Torch-Free App Runtime & Memory Checkpoint
- **Date**: 2026-10-22
- **Refactored `app.py` & Runtime Lifecycle**:
  - `app.py` refactored to use `src/runtime/` exclusively.
  - Implemented `src/runtime/gate.py` (torch-free, sklearn-free, scipy-free via built-in `math.erf` for normal CDF).
  - Implemented `src/runtime/memory.py` (replays events over `RuntimeScorer`).
  - Added `__call__` alias to `OnnxFeatureExtractor`.
  - Added historical model label: `"v1.0 historical / exploratory — model selection leaked test data; not a validated result"`.
- **Subprocess Clean-Import Verification**:
  - `python -c "import app; import sys; assert 'torch' not in sys.modules; assert 'torchvision' not in sys.modules; assert 'sklearn' not in sys.modules"` -> PASSED (exit code 0).
- **Memory Footprint Profile (`reports/app_mvp/app_memory_profile.json`)**:
  - Script: `scripts/measure_app.py`
  - Base RSS (after importing app): `204.43 MB`
  - Post Model Load RSS: `204.43 MB`
  - Post 5-Shot Enrollment RSS: `291.96 MB`
  - Peak RSS: `299.94 MB`
  - Memory Headroom (512 MB - Peak): `212.06 MB`
  - Acceptance threshold: Peak RSS <= `420.0 MB` -> **PASSED** (70.1 MB buffer below limit, 212.06 MB headroom).
  - Torch / Torchvision / Sklearn Leaked: **False** (verified at every inspection & enrollment step).
- **Tests Added & Suite Status**:
  - Added `test_app_imports_no_torch_or_sklearn` to `tests/test_runtime.py`.
  - Full test suite: `68 passed in 116.22s` (0 failures, 0 warnings).
- **Status**: PASSED S2.

## Stage S3: Enrollment Safety Rule & Operator Review Flow
- **Date**: 2026-10-22
- **Strict Enrollment Policy Implementation (`app.py` & `src/runtime/memory.py`)**:
  - `REJECT`: strictly blocked with clear error notes, zero writes to memory log or database.
  - `REVIEW`: requires operator justification note (min 10 characters) explaining why warning was accepted. If note is blank or <10 chars, blocked with validation message. If provided, stored in `gate_report["operator_note"]` and the persisted event.
  - `ACCEPT`: clean enrollment without requiring a review note.
  - Added `operator_note_input` textbox to Gradio Operator UI tab.
  - Fixed `c_hat` closed-set decision argmin in `app.py` to evaluate over all active classes (known base + enrolled classes).
- **Verification on Held-Out Sc**:
  - 5-shot enrollment of Sc using images from `held_enroll_pool` exercised successfully.
  - Gate evaluated novelty, consistency, and overlap; review note correctly captured.
  - Post-enrollment inspection on query Sc image recognizes enrolled defect class (`c_hat` = `Sc_scratches`, distance 3372.0 vs next closest 6030.4).
- **Unit Tests Added (`tests/test_app.py`)**:
  - `test_reject_safety_rule_blocks_invalid_enrollments`: verifies blocking on <5 images, empty name, existing class name, and duplicate images (REJECT outcome). Zero events written.
  - `test_review_safety_rule_requires_operator_justification_note`: verifies blocking when note <10 chars, and successful enrollment + note preservation when >=10 chars.
  - `test_sc_enrollment_and_post_enrollment_recognition`: verifies end-to-end enrollment and post-enrollment inspection verdict containing enrolled class name.
  - Test suite status: `tests/test_app.py` (4/4 passed), `tests/test_runtime.py` (4/4 passed).
- **Status**: PASSED S3.

