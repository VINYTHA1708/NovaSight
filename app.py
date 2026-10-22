"""NovaSight — Manufacturing Open-Set Steel Defect Inspection Platform.

Faculty Review 0 MVP — Industrial Web Application.
Operates fully torch-free and sklearn-free at runtime via ONNX Runtime & float64 NumPy.

Evaluation Status:
"v1.0 historical / exploratory — model selection leaked test data; not a validated result"
"""
from __future__ import annotations

import collections
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logging import setup_logging
from src.config import get_config, REPO_ROOT
from src.utils.hashing import sha256_bytes
from src.runtime import (
    load_runtime_artifact,
    RuntimeArtifact,
    RuntimeScorer,
    OnnxFeatureExtractor,
    preprocess_image,
    load_and_check_app_image_numpy,
    SanityCheckError,
    run_gate,
    GateReport,
    RuntimeMemory,
    replay_events,
    make_decision,
)
from src.app.storage import get_storage, StorageBackend
from src.app.stream import ProductionStream

logger = logging.getLogger(__name__)
setup_logging("WARNING")

HISTORICAL_DISCLAIMER = "v1.0 historical / exploratory — model selection leaked test data; not a validated result"

# ── Runtime & Storage Initialization ──────────────────────────────────────────

try:
    ARTIFACT = load_runtime_artifact()
    SCORER = RuntimeScorer(ARTIFACT)
    ONNX_PATH = REPO_ROOT / "models" / "onnx" / "resnet18_b0_l4.onnx"
    EXTRACTOR = OnnxFeatureExtractor(ONNX_PATH)
    ARTIFACT_LOADED = True
    LOAD_ERROR = ""
except Exception as e:
    ARTIFACT = SCORER = EXTRACTOR = None
    ARTIFACT_LOADED = False
    LOAD_ERROR = str(e)

cfg = get_config()
EVENTS_PATH = REPO_ROOT / cfg["paths"]["state"] / "events.jsonl"
STORAGE: StorageBackend = get_storage()


def get_memory() -> RuntimeMemory:
    """Rebuild memory state from active database storage."""
    return STORAGE.rebuild_memory(SCORER, "H1")


STREAM: ProductionStream | None = None
if ARTIFACT_LOADED:
    STREAM = ProductionStream(
        storage=STORAGE,
        scorer=SCORER,
        extractor=EXTRACTOR,
        get_memory_fn=get_memory,
        speed_fps=1.0,
    )

# Staging bucket for 5-shot enrollment from operator review
STAGING_BUCKET: list[dict[str, Any]] = []


# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_feature_from_pil(img: Image.Image) -> np.ndarray:
    """Extract (512,) feature vector from PIL image via ONNX runtime."""
    inp_tensor = preprocess_image(img)
    return EXTRACTOR(inp_tensor)


# ── Tab 1: Real-Time Inspector ────────────────────────────────────────────────

def inspect_image(image, alpha_value):
    """Process an uploaded image and return a decision + score chart."""
    if not ARTIFACT_LOADED:
        return f"ERROR: Artifact not loaded. {LOAD_ERROR}", None, "{}", False

    if image is None:
        return "No image uploaded.", None, "{}", False

    t0 = time.perf_counter()

    try:
        img_pil = Image.fromarray(image.astype("uint8"), "RGB")
        img_gray = load_and_check_app_image_numpy(img_pil)
    except SanityCheckError as e:
        return f"⚠ Rejected: {e}", None, "{}", False
    except Exception as e:
        img_pil = Image.fromarray(image.astype("uint8"), "RGB")
        img_gray = img_pil.convert("L")

    feat = extract_feature_from_pil(img_gray)
    mem = get_memory()
    alpha = float(alpha_value)

    pred_set, all_p_vals = mem.full_prediction_set(feat, alpha)
    scores = {c: SCORER.inflated_score(feat, ci) for ci, c in enumerate(SCORER.class_names)}
    for name, cls in mem.enrolled.items():
        scores[name] = cls.inflated_score(feat, SCORER.A, SCORER.artifact.tau_hat)

    c_hat = min(scores.keys(), key=lambda c: scores[c])
    decision = make_decision(pred_set, c_hat, list(mem.enrolled.keys()))

    closest_text = f"Closest class: **{c_hat}** (lowest distance: {scores[c_hat]:.1f})"
    if decision["verdict"] == "known":
        verdict = (
            f"✅ **Known Defect: {decision['class_name']}**\n\n"
            f"Conformal set membership: `{decision['class_name']} ∈ Γ_{alpha}` (p = {all_p_vals.get(decision['class_name'], 0.0):.4f})\n\n"
            f"{closest_text}"
        )
        flag_op = False
    elif decision["verdict"] == "unknown":
        verdict = (
            f"🚨 **Unknown Defect — Flagged for Operator Queue**\n\n"
            f"Prediction set empty: `Γ_{alpha} = ∅` (all p-values ≤ {alpha})\n\n"
            f"{closest_text}"
        )
        flag_op = True
    else:
        verdict = (
            f"🔶 **Ambiguous Classification: {', '.join(decision['ambiguous_classes'])}**\n\n"
            f"Prediction set contains multiple candidates: `|Γ_{alpha}| = {len(decision['ambiguous_classes'])}`\n\n"
            f"{closest_text}"
        )
        flag_op = True

    details = {
        "closed_set_label": c_hat,
        "novelty_score": float(mem.novelty_score(feat)),
        "per_class_dt": {k: float(v) for k, v in scores.items()},
        "p_values": {k: float(v) for k, v in all_p_vals.items()},
        "prediction_set": sorted(pred_set),
        "enrolled_classes": list(mem.enrolled.keys()),
        "alpha": alpha,
        "model_status": HISTORICAL_DISCLAIMER,
    }

    latency_ms = (time.perf_counter() - t0) * 1000.0
    img_sha = sha256_bytes(image.tobytes()) if hasattr(image, "tobytes") else ""

    STORAGE.record_inspection({
        "image_sha256": img_sha,
        "verdict": decision["verdict"],
        "class_name": decision["class_name"] or c_hat,
        "novelty_score": float(details["novelty_score"]),
        "prediction_set": details["prediction_set"],
        "details": details,
        "flagged_for_op": flag_op,
        "latency_ms": latency_ms,
    })

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#ffffff")
        c_names = list(scores.keys())
        dt_vals = [scores[c] for c in c_names]
        colors = ["#22c55e" if c in pred_set else "#ef4444" for c in c_names]
        y_pos = np.arange(len(c_names))
        ax.barh(y_pos, dt_vals, color=colors, height=0.6, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(c_names, fontsize=9, fontweight="bold")
        ax.set_xlabel("Inflated Mahalanobis Score (lower = more confident)", fontsize=9)
        ax.set_title(f"Distance Profile & Conformal Set (α={alpha})", fontsize=10, fontweight="bold")
        ax.axvline(SCORER.artifact.tau_hat, color="#f59e0b", linestyle="--", linewidth=1.5,
                   label=f"Calibration Threshold τ̂ = {SCORER.artifact.tau_hat:.1f}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        buf.seek(0)
        chart_img = Image.open(buf).copy()
    except Exception as exc:
        logger.warning("Chart failed: %s", exc)
        chart_img = None

    return verdict, chart_img, json.dumps(details, indent=2), flag_op


# ── Tab 2: Production Line Stream ─────────────────────────────────────────────

def stream_start_action():
    if STREAM:
        STREAM.start()
        return "▶ **Line Status: RUNNING**"
    return "Stream unavailable."

def stream_pause_toggle():
    if not STREAM:
        return "Stream unavailable."
    if STREAM.is_paused:
        STREAM.resume()
        return "▶ **Line Status: RUNNING**"
    else:
        STREAM.pause()
        return "⏸ **Line Status: PAUSED**"

def stream_stop_action():
    if STREAM:
        STREAM.stop()
        return "⏹ **Line Status: STOPPED**"
    return "Stream unavailable."

def stream_speed_change(speed):
    if STREAM:
        STREAM.set_speed(speed)
        return f"Speed set to {speed:.1f} img/sec"
    return "Stream unavailable."

def stream_inject_burst():
    if STREAM:
        n = STREAM.inject_defect_burst(5)
        return f"⚡ Injected {n} Sc defect images into the line!"
    return "Stream unavailable."

def stream_poll_ticker():
    """Poll recent inspections and format markdown ticker table."""
    if not STREAM:
        return "Production stream inactive.", "Stopped"
    recent = STREAM.get_recent_inspections(10)
    status_str = "RUNNING" if STREAM.is_running and not STREAM.is_paused else ("PAUSED" if STREAM.is_paused else "STOPPED")

    if not recent:
        return "*Line initialized. Click 'Start Line' to begin real-time inspection stream.*", status_str

    lines = [
        "| ID | Time | Image Source | Verdict | Predicted Class | Novelty | Latency | Flag |",
        "|:---:|:---:|:---|:---:|:---|:---:|:---:|:---:|",
    ]
    for r in reversed(recent):
        flag_badge = "🚨 **FLAGGED**" if r.get("flagged_for_op") else "✅ OK"
        verdict_badge = f"`{r.get('verdict')}`"
        lines.append(
            f"| {r.get('id', '-')} | {r.get('timestamp', '-')} | `{r.get('image_filename', 'stream')}` | "
            f"{verdict_badge} | **{r.get('class_name', '-')}** | {r.get('novelty_score', 0.0):.1f} | "
            f"{r.get('latency_ms', 0.0):.1f} ms | {flag_badge} |"
        )
    return "\n".join(lines), status_str


# ── Tab 3: Operator Queue ─────────────────────────────────────────────────────

def get_operator_queue_markdown():
    """Fetch unreviewed flagged items from storage."""
    items = STORAGE.list_inspections(limit=30, unreviewed_only=True)
    if not items:
        return "✅ **Operator Queue Clean**: No pending inspection reviews in backlog.", []
    choices = [f"#{it['id']} - {it['ts']} - Verdict: {it['verdict']} - Closest: {it['class_name']}" for it in items]
    lines = [
        f"### 📋 Pending Backlog: {len(items)} items",
        "| ID | Timestamp | Verdict | Closest Class | Novelty | Image SHA256 |",
        "|:---:|:---:|:---:|:---|:---:|:---|",
    ]
    for it in items:
        lines.append(
            f"| {it['id']} | {it['ts'][:19]} | `{it['verdict']}` | **{it['class_name']}** | "
            f"{it['novelty_score']:.1f} | `{it['image_sha256'][:12]}...` |"
        )
    return "\n".join(lines), choices

def review_queue_action(selected_choice, action_type, confirmed_class, review_note):
    """Handle operator review action (Confirm, Stage for 5-shot, Dismiss)."""
    if not selected_choice:
        return "Please select an inspection from the queue."
    ins_id = int(selected_choice.split(" - ")[0].replace("#", ""))
    note = (review_note or "").strip()

    if action_type == "Confirm Known Class":
        label = confirmed_class or "confirmed_defect"
        full_note = f"Operator confirmed as {label}. {note}".strip()
        STORAGE.update_inspection_review(ins_id, label, full_note)
        return f"✅ Inspection #{ins_id} confirmed as **'{label}'**."
    elif action_type == "Stage for 5-Shot Enrollment":
        STAGING_BUCKET.append({"id": ins_id, "note": note})
        STORAGE.update_inspection_review(ins_id, "staged_for_enrollment", f"Staged for enrollment. {note}".strip())
        return f"📦 Inspection #{ins_id} staged! Staging bucket count: {len(STAGING_BUCKET)}/5."
    else:  # Dismiss
        STORAGE.update_inspection_review(ins_id, "dismissed", f"Dismissed by operator. {note}".strip())
        return f"🗑 Inspection #{ins_id} dismissed."

def get_staging_status_markdown():
    n = len(STAGING_BUCKET)
    return f"**Staging Bucket:** `{n}/5` images staged for rapid 5-shot enrollment."


# ── Tab 4: Enroll New Defect Class ───────────────────────────────────────────

def enroll_new_class(
    class_name,
    img1, img2, img3, img4, img5,
    operator_note="",
):
    """Enroll a new class with the strict safety gate policy."""
    if not ARTIFACT_LOADED:
        return f"ERROR: Artifact not loaded. {LOAD_ERROR}", ""

    images = [img1, img2, img3, img4, img5]
    if any(img is None for img in images):
        return "⛔ Enrollment blocked: Please upload exactly 5 images.", ""
    if not class_name or not class_name.strip():
        return "⛔ Enrollment blocked: Please enter a class name.", ""

    mem = get_memory()
    if class_name in mem.all_class_names:
        return f"⛔ Enrollment blocked: Class '{class_name}' already exists. Use a different name.", ""

    embeddings = []
    sha256s = []
    for i, img in enumerate(images):
        try:
            pil_img = Image.fromarray(img.astype("uint8"), "RGB")
            gray = load_and_check_app_image_numpy(pil_img)
        except SanityCheckError as e:
            return f"⛔ Enrollment blocked: Image {i+1} failed sanity check: {e}", ""
        feat = extract_feature_from_pil(gray)
        embeddings.append(feat)
        sha256s.append(sha256_bytes(img.tobytes()))

    embeddings_arr = np.array(embeddings)

    gate_report = run_gate(
        embeddings=embeddings_arr,
        sha256s=sha256s,
        class_name=class_name,
        existing_class_names=mem.all_class_names,
        scorer=SCORER,
        alpha_cons=cfg["gate"]["alpha_cons_grid"][1],
        q_BH=cfg["gate"]["q_BH_grid"][0],
        eps_max=cfg["gate"]["eps_max_grid"][1],
    )

    gate_text = (
        f"Gate outcome: **{gate_report.outcome}**\n\n"
        f"{gate_report.notes}"
    )

    if gate_report.outcome == "REJECT":
        status = f"⛔ Enrollment blocked by safety gate (REJECT):\n{gate_report.notes}"
        return status, gate_text

    if gate_report.outcome == "REVIEW":
        clean_note = (operator_note or "").strip()
        if len(clean_note) < 10:
            status = (
                f"⚠ Gate flagged warnings (REVIEW). Operator justification note (min 10 characters) "
                f"is required to proceed. Provided length: {len(clean_note)} chars."
            )
            return status, gate_text

        mem_fresh = get_memory()
        cls = mem_fresh.enroll(
            name=class_name,
            embeddings=embeddings_arr,
            sha256s=sha256s,
            gate_report=gate_report.to_dict(),
            operator_id="operator",
            operator_note=clean_note,
        )
        STORAGE.save_class(cls)
        STORAGE.record_event({
            "type": "enroll",
            "class_name": class_name,
            "n": 5,
            "sha256s": sha256s,
            "r_hat": cls.r_hat,
            "gate_report": gate_report.to_dict(),
            "operator_id": "operator",
            "operator_note": clean_note,
        })
        status = f"⚠ Class **'{class_name}'** enrolled with confirmed operator review note: \"{clean_note}\"."
        return status, gate_text

    # ACCEPT
    mem_fresh = get_memory()
    cls = mem_fresh.enroll(
        name=class_name,
        embeddings=embeddings_arr,
        sha256s=sha256s,
        gate_report=gate_report.to_dict(),
        operator_id="operator",
    )
    STORAGE.save_class(cls)
    STORAGE.record_event({
        "type": "enroll",
        "class_name": class_name,
        "n": 5,
        "sha256s": sha256s,
        "r_hat": cls.r_hat,
        "gate_report": gate_report.to_dict(),
        "operator_id": "operator",
    })
    status = f"✅ Class **'{class_name}'** enrolled successfully (Gate: ACCEPT)."
    return status, gate_text


def list_enrolled_classes():
    """Return a formatted table of enrolled classes."""
    classes = STORAGE.list_classes()
    if not classes:
        return "No classes enrolled yet."
    lines = ["| Class | n | Inheritance | Gate | Enrolled At |",
             "|-------|---|-------------|------|-------------|"]
    for cls in classes:
        lines.append(
            f"| **{cls.name}** | {cls.n} | {cls.inheritance} | "
            f"`{cls.gate_report.get('outcome', '?')}` | {cls.enrolled_at[:19]} |"
        )
    return "\n".join(lines)


def remove_class(class_name):
    """Remove an enrolled class by name."""
    if not class_name:
        return "Please enter a class name."
    mem = get_memory()
    try:
        mem.remove(class_name)
        STORAGE.delete_class(class_name)
        STORAGE.record_event({"type": "remove", "class_name": class_name, "operator_id": "operator"})
        return f"Removed class '{class_name}'."
    except ValueError as e:
        return str(e)


# ── Tab 5: Records & Statistics ───────────────────────────────────────────────

def get_stats_cards_markdown():
    """Live SQL KPI aggregations."""
    stats = STORAGE.get_inspection_stats()
    total = stats.get("total_inspections", 0)
    flagged = stats.get("flagged_count", 0)
    unreviewed = stats.get("unreviewed_queue_count", 0)
    reviewed = stats.get("reviewed_count", 0)
    avg_lat = stats.get("avg_latency_ms", 0.0)
    flagged_rate = (flagged / total * 100.0) if total > 0 else 0.0

    md = f"""
| Total Inspections | Flagged Count | Flagged Rate | Review Backlog | Operator Reviewed | Avg Latency |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **{total:,}** | **{flagged:,}** | **{flagged_rate:.1f}%** | <span style="color:#ef4444;font-weight:bold;">{unreviewed}</span> | **{reviewed}** | **{avg_lat:.1f} ms** |
"""
    return md

def get_stats_plots():
    """Render live verdict and class distributions."""
    stats = STORAGE.get_inspection_stats()
    verdicts = stats.get("verdict_distribution", {})
    classes = stats.get("class_distribution", {})

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2), facecolor="#ffffff")

    # Verdicts
    v_labels = list(verdicts.keys()) or ["None"]
    v_counts = [verdicts.get(k, 0) for k in v_labels]
    v_colors = ["#22c55e" if k == "known" else ("#ef4444" if k == "unknown" else "#f59e0b") for k in v_labels]
    ax1.bar(v_labels, v_counts, color=v_colors, width=0.5)
    ax1.set_title("Verdict Distribution", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Classes
    c_labels = list(classes.keys()) or ["None"]
    c_counts = [classes.get(k, 0) for k in c_labels]
    ax2.barh(c_labels, c_counts, color="#3b82f6", height=0.5)
    ax2.set_title("Defect Class Distribution", fontsize=10, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

def get_recent_inspections_table():
    inspections = STORAGE.list_inspections(limit=50)
    if not inspections:
        return "No inspections recorded yet."
    lines = [
        "| ID | Timestamp | Verdict | Class | Novelty | Latency | Flagged | Reviewed |",
        "|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|",
    ]
    for it in inspections:
        flag = "🚨 Yes" if it["flagged_for_op"] else "No"
        rev = "✅ Yes" if it["reviewed"] else "No"
        lines.append(
            f"| {it['id']} | {it['ts'][:19]} | `{it['verdict']}` | **{it['class_name']}** | "
            f"{it['novelty_score']:.1f} | {it['latency_ms']:.1f} ms | {flag} | {rev} |"
        )
    return "\n".join(lines)

def export_events_jsonl_file():
    content = STORAGE.export_events_jsonl()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name

def export_events_csv_file():
    content = STORAGE.export_events_csv()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


# ── Tab 6: Model & Evidence (Committed Evidence) ──────────────────────────────

def render_model_evidence_markdown():
    """Display committed historical v1.0 benchmark results without reading test pools."""
    res_path = REPO_ROOT / "reports" / "results_H=Sc_s=0_B0_L4.json"
    card_path = REPO_ROOT / "models" / "onnx" / "MODEL_CARD.json"

    data = {}
    if res_path.exists():
        with open(res_path, encoding="utf-8") as f:
            data = json.load(f)

    card = {}
    if card_path.exists():
        with open(card_path, encoding="utf-8") as f:
            card = json.load(f)

    md = f"""
### ⚠️ Historical v1.0 Model Benchmark Evidence
> **EXPLORATORY RESULT ONLY**: Model selection leaked test-split data during v1.0 exploratory development.  
> Formally superseded by Protocol v1.1 rigorous evaluation.

#### 1. Primary Run Benchmark (H=Sc, Split Seed 0, ResNet18 B0 Layer 4)
- **Known-Class Test Accuracy (Closed-Set)**: `{data.get('T1', {}).get('overall_accuracy', 1.0) * 100:.1f}%` (200 / 200 correct)
- **Open-Set Novelty AUROC**: `{data.get('T2', {}).get('AUROC_novasight', 0.9829):.4f}`
- **Known-Class False Rejection Rate (α=0.05)**: `{data.get('T3', {}).get('known_FRR', 0.015) * 100:.1f}%` (n = 200)
- **Unknown Defect Rejection Rate (α=0.05)**: `{data.get('T3', {}).get('unk_rejection_rate', 0.888) * 100:.1f}%` (n = 125 Sc images)
- **5-Shot Enrollment Mean Recall (H1, 10 draws)**: `{data.get('T5', {}).get('mean_recall_H1', 0.1296) * 100:.2f}%`

#### 2. Committed Historical Confusion Matrix (Test Set)
| True Class \\ Predicted | Crazing | Inclusion | Patches | Pitted Surface | Rolled-in Scale | Unknown Flagged (Γ_α=∅) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Crazing (Cr, n=40)** | **39** | 0 | 0 | 0 | 0 | 1 |
| **Inclusion (In, n=40)** | 0 | **40** | 0 | 0 | 0 | 0 |
| **Patches (Pa, n=40)** | 0 | 0 | **39** | 0 | 0 | 1 |
| **Pitted Surface (PS, n=40)** | 0 | 0 | 0 | **39** | 0 | 1 |
| **Rolled-in Scale (RS, n=40)** | 0 | 0 | 0 | 0 | **40** | 0 |
| **Scratches (Sc, n=125)** | 3 | 5 | 2 | 2 | 2 | **111 (88.8%)** |

#### 3. ONNX Model Provenance
- **Model Path**: `models/onnx/resnet18_b0_l4.onnx`
- **Architecture**: ResNet18 (PyTorch B0 L4 equivalent)
- **Input Dimensions**: `(1, 3, 224, 224)` Float32 (ImageNet normalized)
- **Output Dimensions**: `(512,)` Float64 feature vector
- **ONNX File SHA-256**: `{card.get('model_sha256', '388c2afc2a63797524d93f0c89469771ad1c11e96e4b08bcb813e0a2a134ff4c')}`
"""
    return md


# ── Gradio UI Definition ──────────────────────────────────────────────────────

CSS = """
.verdict-box { font-size: 1.25em; padding: 14px; border-radius: 8px; border: 1px solid #e2e8f0; }
.disclaimer-banner {
    padding: 12px 16px;
    border-radius: 8px;
    background: #fef3c7;
    border: 1px solid #f59e0b;
    color: #92400e;
    font-size: 0.9em;
    margin-bottom: 12px;
}
.kpi-container { font-size: 1.1em; }
"""

with gr.Blocks(
    title="NovaSight — Manufacturing Open-Set Steel Defect Platform",
) as demo:

    gr.Markdown(f"""
# 🏭 NovaSight — Manufacturing Open-Set Steel Defect Platform
**British Council / NexAI Hack 2026** · Production Runtime (Faculty Review 0 MVP)

<div class="disclaimer-banner">
<strong>Evaluation Status:</strong> {HISTORICAL_DISCLAIMER}<br/>
<strong>Active Storage Provider:</strong> {STORAGE.get_provider_name()} · <strong>Runtime:</strong> Standalone ONNX + Mondrian Conformal Engine
</div>
    """)

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    with gr.Tab("🔍 Real-Time Inspector"):
        with gr.Row():
            with gr.Column(scale=1):
                inp_image = gr.Image(label="Upload Steel Surface Image", type="numpy", image_mode="RGB")
                alpha_slider = gr.Slider(
                    minimum=0.01, maximum=0.20, value=0.05, step=0.01,
                    label="Significance level α (lower = stricter novelty detection)",
                )
                inspect_btn = gr.Button("Inspect Surface", variant="primary", elem_id="inspect_btn")

                examples_dir = REPO_ROOT / "examples"
                demo_examples = []
                for fn in ["Cr_247.bmp", "In_247.bmp", "Pa_247.bmp", "Sc_192.bmp"]:
                    p = examples_dir / fn
                    if p.exists():
                        demo_examples.append([str(p), 0.05])
                if demo_examples:
                    gr.Examples(
                        examples=demo_examples,
                        inputs=[inp_image, alpha_slider],
                        label="Sample Test Surfaces (NEU-CLS, Song & Yan 2013)",
                    )
            with gr.Column(scale=2):
                verdict_out = gr.Markdown(value="*Upload an image and click Inspect Surface.*", elem_classes=["verdict-box"])
                chart_out = gr.Image(label="Inflated Mahalanobis Distance vs Calibration Threshold", type="pil")
                details_out = gr.Code(label="Score & Conformal Details (JSON)", language="json")
                flag_op = gr.Checkbox(label="Flag for Operator Panel", visible=False)

        inspect_btn.click(
            fn=inspect_image,
            inputs=[inp_image, alpha_slider],
            outputs=[verdict_out, chart_out, details_out, flag_op],
        )

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
    with gr.Tab("🏭 Production Line"):
        gr.Markdown("""
### High-Speed Continuous Surface Inspection
Simulates automated rolling mill line inspection. Evaluates surface images continuously in background thread without locking UI.
        """)
        with gr.Row():
            line_status_md = gr.Markdown("⏹ **Line Status: STOPPED**")
            start_line_btn = gr.Button("▶ Start Line", variant="primary")
            pause_line_btn = gr.Button("⏸ Pause / Resume Line")
            stop_line_btn = gr.Button("⏹ Stop Line", variant="stop")
            burst_btn = gr.Button("⚡ Inject 5 Sc Defect Burst", variant="secondary")

        with gr.Row():
            speed_slider = gr.Slider(minimum=0.5, maximum=5.0, value=1.0, step=0.5, label="Line Speed (images/second)")

        ticker_table = gr.Markdown("*Line idle. Click 'Start Line' to begin.*")
        stream_timer = gr.Timer(value=1.5, active=True)

        start_line_btn.click(fn=stream_start_action, inputs=[], outputs=[line_status_md])
        pause_line_btn.click(fn=stream_pause_toggle, inputs=[], outputs=[line_status_md])
        stop_line_btn.click(fn=stream_stop_action, inputs=[], outputs=[line_status_md])
        speed_slider.change(fn=stream_speed_change, inputs=[speed_slider], outputs=[])
        burst_btn.click(fn=stream_inject_burst, inputs=[], outputs=[line_status_md])
        stream_timer.tick(fn=stream_poll_ticker, inputs=[], outputs=[ticker_table, line_status_md])

    # ── TAB 3 ─────────────────────────────────────────────────────────────────
    with gr.Tab("🧑‍🏭 Operator Queue"):
        gr.Markdown("### Operator Review Queue (Pending Flagged Defects)")
        refresh_queue_btn = gr.Button("🔄 Refresh Backlog", elem_id="refresh_queue_btn")
        queue_table_md = gr.Markdown("Loading queue...")

        with gr.Row():
            with gr.Column(scale=1):
                queue_select = gr.Dropdown(label="Select Inspection to Review", choices=[])
                action_choice = gr.Radio(
                    label="Review Action",
                    choices=["Confirm Known Class", "Stage for 5-Shot Enrollment", "Dismiss as Noise"],
                    value="Confirm Known Class",
                )
                confirmed_class_input = gr.Textbox(label="Confirmed Class Name", placeholder="e.g., scratches")
                review_note_input = gr.Textbox(label="Review Notes", placeholder="Operator observation...", lines=2)
                submit_review_btn = gr.Button("Submit Operator Decision", variant="primary")
                review_action_status = gr.Markdown()
                staging_status_md = gr.Markdown("**Staging Bucket:** `0/5` staged.")

        refresh_queue_btn.click(fn=get_operator_queue_markdown, inputs=[], outputs=[queue_table_md, queue_select])
        submit_review_btn.click(
            fn=review_queue_action,
            inputs=[queue_select, action_choice, confirmed_class_input, review_note_input],
            outputs=[review_action_status],
        ).then(fn=get_operator_queue_markdown, inputs=[], outputs=[queue_table_md, queue_select]
        ).then(fn=get_staging_status_markdown, inputs=[], outputs=[staging_status_md])

    # ── TAB 4 ─────────────────────────────────────────────────────────────────
    with gr.Tab("➕ Enroll New Defect"):
        gr.Markdown("""
### Five-Shot Rapid Defect Enrollment (No Retraining)
Upload **exactly 5** representative defect images.  
Safety Policy: `REJECT` strictly blocks; `REVIEW` requires operator justification note (≥10 chars).
        """)
        new_class_name = gr.Textbox(label="New Defect Class Name", placeholder="e.g., edge_seam")
        with gr.Row():
            op_img1 = gr.Image(label="Exemplar 1", type="numpy", image_mode="RGB")
            op_img2 = gr.Image(label="Exemplar 2", type="numpy", image_mode="RGB")
            op_img3 = gr.Image(label="Exemplar 3", type="numpy", image_mode="RGB")
            op_img4 = gr.Image(label="Exemplar 4", type="numpy", image_mode="RGB")
            op_img5 = gr.Image(label="Exemplar 5", type="numpy", image_mode="RGB")
        operator_note_input = gr.Textbox(
            label="Operator Justification Note (Required if Gate flags REVIEW, min 10 chars)",
            placeholder="e.g., Confirmed distinct longitudinal surface scratches from mill stand 3",
            lines=2,
        )
        enroll_btn = gr.Button("Run Safety Gate & Enroll Class", variant="primary", elem_id="enroll_btn")
        enroll_status = gr.Markdown()
        gate_details = gr.Markdown(label="Gate Report")

        enroll_btn.click(
            fn=enroll_new_class,
            inputs=[new_class_name, op_img1, op_img2, op_img3, op_img4, op_img5, operator_note_input],
            outputs=[enroll_status, gate_details],
        )

        gr.Markdown("---")
        gr.Markdown("### Currently Enrolled Classes in Memory")
        refresh_mem_btn = gr.Button("Refresh Enrolled List", elem_id="refresh_mem_btn")
        enrolled_table = gr.Markdown()
        refresh_mem_btn.click(fn=list_enrolled_classes, inputs=[], outputs=[enrolled_table])

        with gr.Row():
            remove_name = gr.Textbox(label="Class name to remove")
            remove_btn = gr.Button("Remove Class", variant="stop")
            remove_status = gr.Markdown()
            remove_btn.click(fn=remove_class, inputs=[remove_name], outputs=[remove_status])

    # ── TAB 5 ─────────────────────────────────────────────────────────────────
    with gr.Tab("📊 Records & Statistics"):
        gr.Markdown("### Real-Time Database Metrics & Aggregations")
        refresh_stats_btn = gr.Button("🔄 Refresh Database Statistics", variant="primary")
        kpi_markdown = gr.Markdown(value=get_stats_cards_markdown)

        with gr.Row():
            chart_distribution = gr.Image(label="Live Verdict & Defect Class Distributions", type="pil")

        gr.Markdown("### Inspection Audit Log (Recent 50)")
        recent_log_table = gr.Markdown(value=get_recent_inspections_table)

        gr.Markdown("### Export Event Log")
        with gr.Row():
            export_jsonl_btn = gr.Button("Export Events (JSONL)", elem_id="export_jsonl_btn")
            export_csv_btn = gr.Button("Export Events (CSV)", elem_id="export_csv_btn")
        export_download = gr.File(label="Download Event Log Export")

        refresh_stats_btn.click(fn=get_stats_cards_markdown, inputs=[], outputs=[kpi_markdown])
        refresh_stats_btn.click(fn=get_stats_plots, inputs=[], outputs=[chart_distribution])
        refresh_stats_btn.click(fn=get_recent_inspections_table, inputs=[], outputs=[recent_log_table])
        export_jsonl_btn.click(fn=export_events_jsonl_file, inputs=[], outputs=[export_download])
        export_csv_btn.click(fn=export_events_csv_file, inputs=[], outputs=[export_download])

    # ── TAB 6 ─────────────────────────────────────────────────────────────────
    with gr.Tab("🔬 Model & Evidence"):
        evidence_md = gr.Markdown(value=render_model_evidence_markdown)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="teal"),
        css=CSS,
    )
