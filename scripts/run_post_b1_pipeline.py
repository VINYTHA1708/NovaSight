"""Post-B1 pipeline: re-run selection with B1 features, build artifact, evaluate.

Run after B1 features are fully cached (known + held-out pools).

Usage:
    .venv/Scripts/python.exe scripts/run_post_b1_pipeline.py
"""
import sys, json, subprocess, logging
from pathlib import Path
sys.path.insert(0, '.')
from src.utils.logging import setup_logging
from src.config import REPO_ROOT

setup_logging("INFO")
logger = logging.getLogger(__name__)

HELD_OUT = "Sc"
SEED = 0
PYTHON = sys.executable

def run(cmd, desc):
    logger.info("Running: %s", desc)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("FAILED: %s (exit %d)", desc, result.returncode)
        sys.exit(1)
    logger.info("OK: %s", desc)

# 1. Delete stale selection result so it re-runs
sel_path = REPO_ROOT / "reports" / "selection_result.json"
if sel_path.exists():
    sel_path.unlink()
    logger.info("Deleted stale selection_result.json")

# 2. Re-run selection protocol (now with B1 features available)
run([PYTHON, "scripts/select_model.py", "--held-out", HELD_OUT, "--seed", str(SEED)],
    "Selection protocol (B0 + B1)")

# 3. Read the selected backbone/feature/k
with open(sel_path) as f:
    sel = json.load(f)
best = sel["selected"]
backbone = best["backbone"]
feature = best["feature"]
k = best["k"]
alpha = best["alpha"]
logger.info("Selected: backbone=%s, feature=%s, k=%d, alpha=%.2f", backbone, feature, k, alpha)

# 4. Build artifact with selected configuration
run([PYTHON, "scripts/build_artifact.py",
     "--backbone", backbone, "--feature", feature,
     "--held-out", HELD_OUT, "--seed", str(SEED), "--k", str(k)],
    f"Build artifact ({backbone}/{feature}, k={k})")

# 5. Run evaluation with selected backbone
run([PYTHON, "scripts/evaluate.py",
     "--backbone", backbone, "--feature", feature,
     "--held-out", HELD_OUT, "--seed", str(SEED), "--alpha", str(alpha)],
    f"Evaluate ({backbone}/{feature}, alpha={alpha})")

# 6. Generate confusion matrix (M2)
run([PYTHON, "scripts/gen_confusion_matrix.py",
     "--backbone", backbone, "--feature", feature,
     "--held-out", HELD_OUT, "--seed", str(SEED), "--alpha", str(alpha)],
    "Confusion matrix (M2)")

logger.info("Post-B1 pipeline complete.")
