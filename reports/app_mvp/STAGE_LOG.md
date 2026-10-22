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

## Stage S4: SQLite / Postgres Repository Layer
- **Date**: 2026-10-22
- **Storage Abstraction (`src/app/storage.py`)**:
  - Implemented `StorageBackend` abstract class with concrete `SQLiteStorage` (default at `state/novasight.db`) and `PostgresStorage` (when `DATABASE_URL` is set).
  - Idempotent schema migrations with 3 tables: `classes`, `events`, and `inspections`.
  - Added methods for saving/listing/deleting classes, recording events, export to JSONL/CSV, recording inspections, updating reviews, and computing aggregate stats.
  - Added `rebuild_memory` method to reconstruct `RuntimeMemory` from database.
- **Application Integration (`app.py`)**:
  - `STORAGE = get_storage()` initialized on startup.
  - Memory rebuilt from database: `get_memory()` calls `STORAGE.rebuild_memory(SCORER, "H1")`.
  - All enrollments, removals, and inspection evaluations persist directly to the database.
  - UI header banner displays active storage provider (`Storage: SQLite (local dev)` or `Storage: Neon Postgres (production)`).
  - Memory tab includes buttons to export memory events to downloadable JSONL and CSV files.
- **Tests Added (`tests/test_storage.py`)**:
  - `test_sqlite_class_crud`: insert class, get class, list classes, delete class.
  - `test_schema_migration_idempotence`: repeated schema migrations cause no data loss.
  - `test_inspections_and_review_workflow`: record inspection, list unreviewed queue, update review, compute statistics.
  - `test_rebuild_memory_from_db`: rebuilds memory matching simulated sequence.
  - `test_export_events_jsonl_and_csv`: verifies both export formats.
  - `test_postgres_storage_live`: clean skipif when `DATABASE_URL` is unset.
  - Test suite status: `5 passed, 1 skipped` in `test_storage.py`; `4 passed` in `test_app.py`.
- **Status**: PASSED S4.

## Stage S5: Inspection Logging, Operator Queue & Live Statistics
- **Date**: 2026-10-22
- **Operator Review Queue (`app.py` & `src/app/storage.py`)**:
  - Unreviewed flagged inspections listable and manageable with review actions: Confirm Known Class, Stage for 5-Shot Enrollment, or Dismiss.
  - Review action updates database record (`reviewed=TRUE`, `review_label`, `review_note`).
  - Added staging bucket for rapid accumulation of 5 novel defect exemplars.
- **Live Database-Derived Statistics**:
  - Total inspection counter, flagged count and rate (%), backlog queue count, reviewed count, mean latency (ms), and defect class & verdict distributions rendered from SQL queries.
- **Tests Added (`tests/test_storage.py`)**:
  - `test_stats_aggregation_50_synthetic_sequence`: verified exact matching aggregation metrics across 50 inspections.
- **Status**: PASSED S5.

## Stage S6: Simulated Production Line Stream
- **Date**: 2026-10-22
- **Production Line State Machine (`src/app/stream.py`)**:
  - Line speed configurable from 0.5 to 5.0 images/sec (default 1.0).
  - Allowed image mix: 85% known-class images (`sel_eval` pool) + 15% novel defect images (`held_enroll_pool` Sc). Strictly zero reads from test/cal pools.
  - Full runtime processing pipeline per tick (preprocess -> ONNX -> scoring -> conformal -> decision).
  - Background execution using `threading.Thread` and `threading.Event` without locking the Gradio interface.
  - Implemented 5-image Sc defect burst injection (`inject_defect_burst`).
- **Tests Added (`tests/test_stream.py`)**:
  - `test_stream_state_machine`: start -> 10 inspections -> pause -> check DB -> resume -> 5 more -> stop.
  - `test_defect_burst_injection`: verified burst queue processing and attribution.
- **Status**: PASSED S6.

## Stage S7: 6-Tab Manufacturing UI Redesign
- **Date**: 2026-10-22
- **Industrial Dashboard Implementation (`app.py`)**:
  - Tab 1: 🔍 **Real-Time Inspector** — Single-surface upload, significance slider, visual conformal membership, Mahalanobis distance chart.
  - Tab 2: 🏭 **Production Line** — Start/Pause/Stop stream controls, live speed slider, defect burst injection, and `gr.Timer` live ticker table.
  - Tab 3: 🧑‍🏭 **Operator Queue** — Unreviewed backlog queue, decision confirmation, 5-shot staging bucket, dismiss workflow.
  - Tab 4: ➕ **Enroll New Defect** — 5-shot enrollment with strict safety gate, operator justification note for REVIEW, enrolled class table & remove.
  - Tab 5: 📊 **Records & Statistics** — Live KPI cards, verdict & class distribution plots, 50-inspection audit log, JSONL & CSV event download.
  - Tab 6: 🔬 **Model & Evidence** — Benchmark evidence, confusion matrix, provenance details, and prominent historical status banner.
- **Zero Leakage**:
  - Verified `torch`, `torchvision`, and `sklearn` remain completely absent from `sys.modules`.
- **Status**: PASSED S7.

## Stage S8: Model & Evidence Tab (Committed Evidence)
- **Date**: 2026-10-22
- **Historical Benchmark Rendering**:
  - Accurately renders committed results from `reports/results_H=Sc_s=0_B0_L4.json` without running runtime inference or accessing test pools.
  - Closed-set accuracy: 100.0% (200/200), Novelty AUROC: 0.9829, Known FRR: 1.5%, Unknown Rejection: 88.8%, 5-shot mean recall: 12.96%.
  - Full 5-known + 1-unknown test confusion matrix rendered directly in markdown.
  - Prominent banner: `"Historical v1.0 Model — Exploratory result with test-data leakage during model selection. Protocol v1.1 evaluation pending."`
- **Status**: PASSED S8.

