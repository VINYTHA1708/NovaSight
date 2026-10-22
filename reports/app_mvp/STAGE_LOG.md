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

