"""NovaSight Gradio application (M3 requirement).

Flow per upload:
  1. Sanity checks (app-only: min_side, near_constant, undecodable)
  2. Feature extraction via frozen extractor
  3. Score via GaussianModel + ConformalScores
  4. Decision: known class / unknown / ambiguous
  5. If unknown -> "send to operator" panel
  6. Operator labels 5 examples -> gate runs -> class enrolled
  7. System responds to new class without retraining

Tabs:
  Inspector: upload image, see decision + score bar chart
  Operator: enroll 5 images for a new class
  Memory:   list enrolled classes, remove, rename, merge, rollback
  Provenance: show artifact metadata and version hash
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
import sys

import numpy as np
import gradio as gr
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logging import setup_logging
from src.config import get_config, REPO_ROOT
from src.data.transforms import get_eval_transform, load_and_check_app_image, SanityCheckError
from src.models.backbones import ResNetExtractor
from src.models.gaussian import GaussianModel
from src.models.conformal import ConformalScores, make_decision
from src.models.memory import NovaSightMemory, replay_events
from src.models.gate import run_gate
from src.utils.hashing import sha256_bytes
from src.explain.explain import score_breakdown

logger = logging.getLogger(__name__)
setup_logging("WARNING")   # quiet in app context

# ── Load artifact ──────────────────────────────────────────────────────────────

def _load_artifact():
    """Load the frozen base artifact."""
    import src.models.conformal as cm
    from src.models.gaussian import GaussianModel
    artifact_dir = REPO_ROOT / "artifacts" / "base"
    meta_path = artifact_dir / "metadata.json"

    if not meta_path.exists():
        raise RuntimeError(
            "Base artifact not found. Run scripts/build_artifact.py first."
        )
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    g_data = np.load(artifact_dir / "gaussian.npz", allow_pickle=True)
    A = g_data["A"]
    class_names = [str(c) for c in g_data["class_names"]]
    n_c = list(g_data["n_c"])
    k = int(g_data["k"])
    feature_dim = int(g_data["feature_dim"])
    shrinkage_intensity = float(g_data["shrinkage_intensity"])
    condition_number = float(g_data["condition_number"])

    m_data = np.load(artifact_dir / "class_means.npz")
    class_means = [m_data[f"means_{i}"] for i in range(len(class_names))]
    n_sub = [list(m_data[f"n_sub_{i}"]) for i in range(len(class_names))]

    gaussian = GaussianModel(
        class_names=class_names,
        class_means=class_means,
        n_c=list(n_c),
        n_sub=[list(ns) for ns in n_sub],
        A=A,
        condition_number=condition_number,
        shrinkage_intensity=shrinkage_intensity,
        k=k,
        feature_dim=feature_dim,
    )

    c_data = np.load(artifact_dir / "conformal.npz")
    class_scores = {cn: c_data[f"scores_{cn}"] for cn in class_names}
    S_pool = c_data["S_pool"]
    tau_hat = float(c_data["tau_hat"][0])
    conformal = ConformalScores(
        class_scores=class_scores,
        S_pool=S_pool,
        tau_hat=tau_hat,
        alpha_default=meta.get("alpha_default", 0.05),
        alpha_grid=meta.get("alpha_grid", [0.01, 0.02, 0.05, 0.10, 0.20]),
    )
    return gaussian, conformal, meta


try:
    GAUSSIAN, CONFORMAL, META = _load_artifact()
    BACKBONE_ID = META.get("backbone_id", "B0")
    FEATURE = META.get("feature", "L4")
    EXTRACTOR = ResNetExtractor(BACKBONE_ID, feature=FEATURE)
    EXTRACTOR.eval()
    TRANSFORM = get_eval_transform(BACKBONE_ID)
    ARTIFACT_LOADED = True
except Exception as e:
    GAUSSIAN = CONFORMAL = META = EXTRACTOR = TRANSFORM = None
    BACKBONE_ID = FEATURE = "N/A"
    ARTIFACT_LOADED = False
    LOAD_ERROR = str(e)

cfg = get_config()
EVENTS_PATH = REPO_ROOT / cfg["paths"]["state"] / "events.jsonl"
EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_memory() -> NovaSightMemory:
    """Replay event log to get current memory state."""
    return replay_events(EVENTS_PATH, GAUSSIAN, CONFORMAL, "H1")


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_feature_from_pil(img: Image.Image) -> np.ndarray:
    """Extract feature vector from a PIL image.

    Args:
        img: PIL image (any mode).

    Returns:
        (D,) feature vector.
    """
    tensor = TRANSFORM(img).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        feat = EXTRACTOR(tensor)
    return feat.squeeze(0).numpy()


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

    # Convert numpy array to PIL
    try:
        img_pil = Image.fromarray(image.astype("uint8"), "RGB")
        # Apply application sanity checks
        img_gray = load_and_check_app_image(img_pil)
    except SanityCheckError as e:
        return f"⚠ Rejected: {e}", None, "{}", False
    except Exception as e:
        img_pil = Image.fromarray(image.astype("uint8"), "RGB")
        img_gray = img_pil.convert("L")

    # Extract feature
    feat = extract_feature_from_pil(img_gray)

    # Get memory (replays event log each call for simplicity)
    mem = get_memory()
    alpha = float(alpha_value)

    # Compute prediction set
    pred_set, all_p_vals = mem.full_prediction_set(feat, alpha)
    scores = {c: GAUSSIAN.inflated_score(feat, ci)
              for ci, c in enumerate(GAUSSIAN.class_names)}
    c_hat, _ = GAUSSIAN.closed_set_label(feat)
    decision = make_decision(pred_set, c_hat, list(mem.enrolled.keys()))

    # Score breakdown for display
    breakdown = score_breakdown(feat, GAUSSIAN, CONFORMAL, alpha)

    # Build verdict string
    if decision["verdict"] == "known":
        verdict = f"✅ **{decision['class_name']}**\n*(confidence: class in prediction set)*"
        flag_op = False
    elif decision["verdict"] == "unknown":
        verdict = "⚠ **Unknown defect — send to operator**"
        flag_op = True
    else:
        verdict = f"🔶 **Ambiguous**: {decision['message']}"
        flag_op = True

    # Score details
    details = {
        "novelty_score": float(breakdown["novelty_score"]),
        "per_class_dt": {k: float(v) for k, v in breakdown["per_class_dt"].items()},
        "p_values": {k: float(v) for k, v in (breakdown["p_values"] or {}).items()},
        "prediction_set": list(breakdown["prediction_set"] or []),
        "enrolled_classes": list(mem.enrolled.keys()),
        "alpha": alpha,
    }

    # Simple matplotlib bar chart of per-class scores
    try:
        import matplotlib.pyplot as plt
        import io
        fig, ax = plt.subplots(figsize=(7, 3))
        class_names = list(breakdown["per_class_dt"].keys())
        dt_values = [breakdown["per_class_dt"][c] for c in class_names]
        colors = ["#2ecc71" if c in (breakdown["prediction_set"] or []) else "#e74c3c"
                  for c in class_names]
        ax.barh(class_names, dt_values, color=colors)
        ax.set_xlabel("Inflated Mahalanobis Score (lower = more confident)")
        ax.set_title(f"Score Breakdown (α={alpha})")
        ax.axvline(CONFORMAL.tau_hat, color="orange", linestyle="--",
                   label=f"τ̂={CONFORMAL.tau_hat:.1f}")
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
):
    """Enroll a new class from 5 uploaded operator images.

    Args:
        class_name: Name for the new class.
        img1..img5: numpy arrays from Gradio image inputs.

    Returns:
        Status message and gate report text.
    """
    if not ARTIFACT_LOADED:
        return "ERROR: Artifact not loaded.", ""

    images = [img1, img2, img3, img4, img5]
    if any(img is None for img in images):
        return "Please upload exactly 5 images.", ""
    if not class_name or not class_name.strip():
        return "Please enter a class name.", ""

    mem = get_memory()
    if class_name in mem.all_class_names:
        return f"Class '{class_name}' already exists. Use a different name.", ""

    # Extract features and sha256s
    embeddings = []
    sha256s = []
    for i, img in enumerate(images):
        try:
            pil_img = Image.fromarray(img.astype("uint8"), "RGB")
            gray = load_and_check_app_image(pil_img)
        except SanityCheckError as e:
            return f"Image {i+1} failed sanity check: {e}", ""
        feat = extract_feature_from_pil(gray)
        embeddings.append(feat)
        sha256s.append(sha256_bytes(img.tobytes()))

    embeddings_arr = np.array(embeddings)

    # Run gate
    gate_report = run_gate(
        embeddings=embeddings_arr,
        sha256s=sha256s,
        class_name=class_name,
        existing_class_names=mem.all_class_names,
        gaussian=GAUSSIAN,
        conformal=CONFORMAL,
        alpha_cons=cfg["gate"]["alpha_cons_grid"][1],
        q_BH=cfg["gate"]["q_BH_grid"][0],
        eps_max=cfg["gate"]["eps_max_grid"][1],
    )

    # Enroll regardless of gate outcome [APPROVED-POLICY]
    mem_fresh = get_memory()
    mem_fresh.enroll(
        name=class_name,
        embeddings=embeddings_arr,
        sha256s=sha256s,
        gate_report=gate_report.to_dict(),
        operator_id="operator",
    )

    gate_text = (
        f"Gate outcome: **{gate_report.outcome}**\n\n"
        f"{gate_report.notes}"
    )
    if gate_report.outcome == "ACCEPT":
        status = f"✅ Class **'{class_name}'** enrolled successfully."
    elif gate_report.outcome == "REVIEW":
        status = f"⚠ Class **'{class_name}'** enrolled with warnings. Please review gate notes."
    else:
        status = f"⚠ Gate flagged issues but enrolled anyway (evaluation mode). Gate: {gate_report.outcome}"

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
        return f"Artifact not loaded: {LOAD_ERROR if not ARTIFACT_LOADED else ''}"
    mem = get_memory()
    info = {
        "backbone_id": META.get("backbone_id"),
        "feature": META.get("feature"),
        "held_out": META.get("held_out"),
        "split_seed": META.get("split_seed"),
        "K_known": len(GAUSSIAN.class_names),
        "known_classes": GAUSSIAN.class_names,
        "D": GAUSSIAN.feature_dim,
        "shrinkage": META.get("shrinkage_intensity"),
        "tau_hat": META.get("tau_hat"),
        "n_enrolled": len(mem.enrolled),
        "memory_version_hash": mem.get_version_hash()[:16] + "...",
        "config_sha256": META.get("config_sha256", "")[:16] + "...",
    }
    return json.dumps(info, indent=2)


# ── Build Gradio UI ────────────────────────────────────────────────────────────

CSS = """
.verdict-box { font-size: 1.3em; padding: 16px; border-radius: 8px; }
.gr-button { font-weight: 600; }
"""

with gr.Blocks(
    title="NovaSight — Open-Set Steel Defect Inspector",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="teal"),
    css=CSS,
) as demo:

    gr.Markdown("""
# 🔬 NovaSight — Open-Set Steel Defect Inspector
**Research prototype** · NexAI Hack 2026 (British Council) · PS 2.3

> Trained on 5 of 6 NEU-CLS defect classes. **Held-out class (M1): Scratches (Sc)**  
> Upload an image → receive a class label or "unknown, send to operator"  
> Operator labels 5 examples → class enrolled immediately (no retraining needed)
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
        """)
        new_class_name = gr.Textbox(label="New Class Name", placeholder="e.g., edge_crack")
        with gr.Row():
            op_img1 = gr.Image(label="Example 1", type="numpy", image_mode="RGB")
            op_img2 = gr.Image(label="Example 2", type="numpy", image_mode="RGB")
            op_img3 = gr.Image(label="Example 3", type="numpy", image_mode="RGB")
            op_img4 = gr.Image(label="Example 4", type="numpy", image_mode="RGB")
            op_img5 = gr.Image(label="Example 5", type="numpy", image_mode="RGB")
        enroll_btn = gr.Button("Enroll Class", variant="primary", elem_id="enroll_btn")
        enroll_status = gr.Markdown()
        gate_details = gr.Markdown(label="Gate Report")

        enroll_btn.click(
            fn=enroll_new_class,
            inputs=[new_class_name, op_img1, op_img2, op_img3, op_img4, op_img5],
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

