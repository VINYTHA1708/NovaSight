"""NovaSight Gradio application (Torch-Free Production Runtime).

Historical note: ResNet18 B0 L4 base model is labeled as
"v1.0 historical / exploratory — model selection leaked test data; not a validated result".

Flow per upload:
  1. Sanity checks (min_side, near_constant, undecodable)
  2. Feature extraction via ONNX runtime (resnet18_b0_l4.onnx)
  3. Score via RuntimeScorer (inflated Mahalanobis + Mondrian conformal)
  4. Decision: known class / unknown / ambiguous
  5. If unknown -> "send to operator" panel
  6. Operator labels 5 examples -> gate checks -> class enrolled immediately
  7. System responds to newly enrolled class without retraining
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

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

logger = logging.getLogger(__name__)
setup_logging("WARNING")

HISTORICAL_DISCLAIMER = "v1.0 historical / exploratory — model selection leaked test data; not a validated result"

# ── Load artifact & ONNX runtime ──────────────────────────────────────────────

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
EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
if EVENTS_PATH.exists():
    try:
        EVENTS_PATH.unlink()
        logger.info("Cleared previous session events on startup.")
    except Exception as exc:
        logger.warning("Could not clear previous session events: %s", exc)


def get_memory() -> RuntimeMemory:
    """Replay event log to get current memory state."""
    return replay_events(EVENTS_PATH, SCORER, "H1")


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_feature_from_pil(img: Image.Image) -> np.ndarray:
    """Extract (512,) feature vector from PIL image via ONNX runtime."""
    inp_tensor = preprocess_image(img)
    return EXTRACTOR(inp_tensor)


# ── Inspector tab ─────────────────────────────────────────────────────────────

def inspect_image(image, alpha_value):
    """Process an uploaded image and return a decision + score chart.

    Args:
        image: numpy array from Gradio image input (HxWx3 RGB).
        alpha_value: Significance level slider value.

    Returns:
        Tuple of (verdict_text, score_chart_image, details_json, flag_for_operator).
    """
    if not ARTIFACT_LOADED:
        return f"ERROR: Artifact not loaded. {LOAD_ERROR}", None, "{}", False

    if image is None:
        return "No image uploaded.", None, "{}", False

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

    closest_text = f"Closest class: **{c_hat}** (lowest distance)"
    if decision["verdict"] == "known":
        verdict = (
            f"✅ **{decision['class_name']}**\n"
            f"*(conformal set membership: {decision['class_name']} ∈ Γ_α)*\n"
            f"{closest_text}"
        )
        flag_op = False
    elif decision["verdict"] == "unknown":
        verdict = (
            f"⚠ **Unknown defect — send to operator**\n"
            f"{closest_text}"
        )
        flag_op = True
    else:
        verdict = (
            f"🔶 **Ambiguous**: {decision['message']}\n"
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

    try:
        import matplotlib.pyplot as plt
        import io
        fig, ax = plt.subplots(figsize=(7, 3))
        c_names = list(scores.keys())
        dt_vals = [scores[c] for c in c_names]
        colors = ["#2ecc71" if c in pred_set else "#e74c3c" for c in c_names]
        ax.barh(c_names, dt_vals, color=colors)
        ax.set_xlabel("Inflated Mahalanobis Score (lower = more confident)")
        ax.set_title(f"Score Breakdown (α={alpha})")
        ax.axvline(SCORER.artifact.tau_hat, color="orange", linestyle="--",
                   label=f"τ̂={SCORER.artifact.tau_hat:.1f}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        chart_img = Image.open(buf).copy()
    except Exception as exc:
        logger.warning("Chart failed: %s", exc)
        chart_img = None

    return verdict, chart_img, json.dumps(details, indent=2), flag_op


# ── Operator tab ──────────────────────────────────────────────────────────────

def enroll_new_class(
    class_name,
    img1, img2, img3, img4, img5,
    operator_note="",
):
    """Enroll a new class from 5 uploaded operator images following the strict safety rule.

    - REJECT: strictly blocked, clear explanation, zero log writes.
    - REVIEW: requires >=10 char operator justification note.
    - ACCEPT: enrolls cleanly.
    """
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
        mem_fresh.enroll(
            name=class_name,
            embeddings=embeddings_arr,
            sha256s=sha256s,
            gate_report=gate_report.to_dict(),
            operator_id="operator",
            operator_note=clean_note,
        )
        status = f"⚠ Class **'{class_name}'** enrolled with confirmed operator review note: \"{clean_note}\"."
        return status, gate_text

    # ACCEPT
    mem_fresh = get_memory()
    mem_fresh.enroll(
        name=class_name,
        embeddings=embeddings_arr,
        sha256s=sha256s,
        gate_report=gate_report.to_dict(),
        operator_id="operator",
    )
    status = f"✅ Class **'{class_name}'** enrolled successfully (Gate: ACCEPT)."
    return status, gate_text


# ── Memory tab ────────────────────────────────────────────────────────────────

def list_enrolled_classes():
    """Return a formatted table of enrolled classes."""
    if not ARTIFACT_LOADED:
        return "Artifact not loaded."
    mem = get_memory()
    if not mem.enrolled:
        return "No classes enrolled yet."
    lines = ["| Class | n | Inheritance | Gate | Enrolled At |",
             "|-------|---|-------------|------|-------------|"]
    for name, cls in mem.enrolled.items():
        lines.append(
            f"| {name} | {cls.n} | {cls.inheritance} | "
            f"{cls.gate_report.get('outcome', '?')} | {cls.enrolled_at[:19]} |"
        )
    return "\n".join(lines)


def remove_class(class_name):
    """Remove an enrolled class by name."""
    if not class_name:
        return "Please enter a class name."
    mem = get_memory()
    try:
        mem.remove(class_name)
        return f"Removed class '{class_name}'."
    except ValueError as e:
        return str(e)


# ── Provenance tab ────────────────────────────────────────────────────────────

def show_provenance():
    """Display artifact metadata and memory version hash."""
    if not ARTIFACT_LOADED:
        return f"Artifact not loaded: {LOAD_ERROR}"
    mem = get_memory()
    info = {
        "model_label": HISTORICAL_DISCLAIMER,
        "backbone_id": ARTIFACT.metadata.get("backbone_id"),
        "feature": ARTIFACT.metadata.get("feature"),
        "held_out": ARTIFACT.metadata.get("held_out"),
        "split_seed": ARTIFACT.metadata.get("split_seed"),
        "K_known": len(SCORER.class_names),
        "known_classes": SCORER.class_names,
        "D": int(ARTIFACT.metadata.get("feature_dim", 512)),
        "shrinkage": ARTIFACT.metadata.get("shrinkage_intensity"),
        "tau_hat": ARTIFACT.tau_hat,
        "n_enrolled": len(mem.enrolled),
        "memory_version_hash": mem.get_version_hash()[:16] + "...",
        "provenance_verified": ARTIFACT.provenance_verified,
    }
    return json.dumps(info, indent=2)


# ── Build Gradio UI ────────────────────────────────────────────────────────────

CSS = """
.verdict-box { font-size: 1.3em; padding: 16px; border-radius: 8px; }
.gr-button { font-weight: 600; }
.disclaimer-banner {
    padding: 10px 14px;
    border-radius: 6px;
    background: #fff3cd;
    border: 1px solid #ffeeba;
    color: #856404;
    font-size: 0.9em;
    margin-bottom: 12px;
}
"""

with gr.Blocks(
    title="NovaSight — Open-Set Steel Defect Inspector",
) as demo:

    gr.Markdown(f"""
# 🔬 NovaSight — Open-Set Steel Defect Inspector
**Research prototype** · NexAI Hack 2026 (British Council) · PS 2.3

<div class="disclaimer-banner">
<strong>Evaluation Status:</strong> {HISTORICAL_DISCLAIMER}
</div>

> Base model trained on 5 of 6 NEU-CLS defect classes. **Held-out class (M1): Scratches (Sc)**  
> Upload an image → receive a class label or "unknown, send to operator"  
> Operator labels 5 examples → class enrolled immediately via pure-runtime inference (no retraining needed)
    """)

    with gr.Tab("🔍 Inspector"):
        with gr.Row():
            with gr.Column(scale=1):
                inp_image = gr.Image(label="Upload Steel Surface Image", type="numpy",
                                     image_mode="RGB")
                alpha_slider = gr.Slider(
                    minimum=0.01, maximum=0.20, value=0.05, step=0.01,
                    label="Significance level α (lower = stricter novelty detection)",
                )
                inspect_btn = gr.Button("Inspect", variant="primary", elem_id="inspect_btn")

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
                        label="Sample Images (NEU-CLS, Song & Yan 2013)",
                    )
            with gr.Column(scale=2):
                verdict_out = gr.Markdown(value="*Upload an image and click Inspect.*",
                                          elem_classes=["verdict-box"])
                chart_out = gr.Image(label="Score Breakdown", type="pil")
                details_out = gr.Code(label="Score Details (JSON)", language="json")
                flag_op = gr.Checkbox(label="Flag for Operator Panel", visible=False)

        inspect_btn.click(
            fn=inspect_image,
            inputs=[inp_image, alpha_slider],
            outputs=[verdict_out, chart_out, details_out, flag_op],
        )

    with gr.Tab("🧑‍🔧 Operator — Enroll New Class"):
        gr.Markdown("""
### Enroll a New Defect Class
Upload **exactly 5** representative images of the new defect type, then click Enroll.  
The system will run gate checks and enroll the class immediately.

*Demo images available in `examples/`: `Sc_189.bmp`, `Sc_148.bmp`, `Sc_101.bmp`, `Sc_13.bmp`, `Sc_29.bmp` (held-out scratches pool; Song & Yan 2013).*
        """)
        new_class_name = gr.Textbox(label="New Class Name", placeholder="e.g., edge_crack")
        with gr.Row():
            op_img1 = gr.Image(label="Example 1", type="numpy", image_mode="RGB")
            op_img2 = gr.Image(label="Example 2", type="numpy", image_mode="RGB")
            op_img3 = gr.Image(label="Example 3", type="numpy", image_mode="RGB")
            op_img4 = gr.Image(label="Example 4", type="numpy", image_mode="RGB")
            op_img5 = gr.Image(label="Example 5", type="numpy", image_mode="RGB")
        operator_note_input = gr.Textbox(
            label="Operator Justification Note (required if Gate outcome is REVIEW, min 10 chars)",
            placeholder="e.g., Confirmed distinct longitudinal surface scratches from rolling line 2",
            lines=2,
        )
        enroll_btn = gr.Button("Enroll Class", variant="primary", elem_id="enroll_btn")
        enroll_status = gr.Markdown()
        gate_details = gr.Markdown(label="Gate Report")

        enroll_btn.click(
            fn=enroll_new_class,
            inputs=[new_class_name, op_img1, op_img2, op_img3, op_img4, op_img5, operator_note_input],
            outputs=[enroll_status, gate_details],
        )

    with gr.Tab("🗂 Memory — Enrolled Classes"):
        gr.Markdown("### Currently enrolled classes (persisted across sessions via event log)")
        refresh_btn = gr.Button("Refresh", elem_id="refresh_btn")
        enrolled_table = gr.Markdown()
        refresh_btn.click(fn=list_enrolled_classes, inputs=[], outputs=[enrolled_table])

        gr.Markdown("### Remove an enrolled class")
        remove_name = gr.Textbox(label="Class name to remove")
        remove_btn = gr.Button("Remove", variant="stop", elem_id="remove_btn")
        remove_status = gr.Markdown()
        remove_btn.click(fn=remove_class, inputs=[remove_name], outputs=[remove_status])

    with gr.Tab("📋 Provenance"):
        gr.Markdown("### Artifact metadata and memory version")
        prov_btn = gr.Button("Show Provenance", elem_id="prov_btn")
        prov_out = gr.Code(language="json")
        prov_btn.click(fn=show_provenance, inputs=[], outputs=[prov_out])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="teal"),
        css=CSS,
    )
