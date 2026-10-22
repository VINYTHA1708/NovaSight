"""Generate 7 high-fidelity UI screens for NovaSight MVP Faculty Review.

Saves figures to reports/app_mvp/screens/:
- 01_inspector_known.png
- 02_inspector_unknown.png
- 03_production_stream.png
- 04_operator_queue.png
- 05_enrollment_gate.png
- 06_records_statistics.png
- 07_model_evidence.png
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

OUT_DIR = Path("reports/app_mvp/screens")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Visual theme constants
BG_COLOR = "#0B0F19"
PANEL_BG = "#131C2E"
CARD_BG = "#1A253C"
TEXT_COLOR = "#F1F5F9"
TEXT_MUTED = "#94A3B8"
ACCENT_BLUE = "#38BDF8"
ACCENT_GREEN = "#22C55E"
ACCENT_RED = "#EF4444"
ACCENT_AMBER = "#F59E0B"
BORDER_COLOR = "#2D3E5E"

plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def create_base_canvas(title: str, subtitle: str, active_tab_idx: int):
    fig = plt.figure(figsize=(14, 8.5), facecolor=BG_COLOR, dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header bar
    header = patches.Rectangle((0, 91), 100, 9, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(header)

    ax.text(3, 96.2, "NOVASIGHT", color=ACCENT_BLUE, fontsize=18, fontweight="bold", va="center")
    ax.text(17, 96.2, "|  Industrial Surface Anomaly & Novel Defect Inspection", color=TEXT_COLOR, fontsize=13, va="center")
    ax.text(3, 93.0, subtitle, color=TEXT_MUTED, fontsize=10, va="center")

    # System Status Indicators in Header
    ax.text(82, 95.8, "RUNTIME: ONNX Engine (Torch-Free)", color=ACCENT_GREEN, fontsize=9, fontweight="bold", ha="right")
    ax.text(82, 93.2, "STORAGE: SQLite (local dev)", color=ACCENT_AMBER, fontsize=9, ha="right")
    ax.text(97, 94.5, "● LIVE", color=ACCENT_GREEN, fontsize=12, fontweight="bold", ha="right")

    # Tab bar
    tabs = [
        "1. Real-Time Inspector",
        "2. Production Line",
        "3. Operator Queue",
        "4. Enroll New Defect",
        "5. Records & Statistics",
        "6. Model & Evidence"
    ]
    tab_w = 15.5
    for i, tab in enumerate(tabs):
        tx = 2 + i * 16.2
        is_active = (i == active_tab_idx)
        t_bg = CARD_BG if is_active else PANEL_BG
        t_edge = ACCENT_BLUE if is_active else BORDER_COLOR
        t_text = ACCENT_BLUE if is_active else TEXT_MUTED
        
        tab_patch = patches.Rectangle((tx, 84.5), tab_w, 5.5, facecolor=t_bg, edgecolor=t_edge, linewidth=1.5 if is_active else 1)
        ax.add_patch(tab_patch)
        ax.text(tx + tab_w/2, 87.2, tab, color=t_text, fontsize=9.5, fontweight="bold" if is_active else "normal", ha="center", va="center")

    return fig, ax


def load_sample_img(path_str):
    p = Path(path_str)
    if p.exists():
        img = Image.open(p).convert("L")
        return np.array(img)
    # Synthetic texture fallback
    x = np.linspace(0, 10, 200)
    y = np.linspace(0, 10, 200)
    xx, yy = np.meshgrid(x, y)
    return (np.sin(xx) * np.cos(yy) * 128 + 128).astype(np.uint8)


def generate_01_inspector_known():
    fig, ax = create_base_canvas(
        "Real-Time Surface Defect Inspector",
        "Single-piece inspection with conformal prediction guarantees (α = 0.05)",
        0
    )

    # Left Panel: Image Viewer
    p_left = patches.Rectangle((3, 6), 46, 75.5, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_left)
    ax.text(5, 78, "INPUT SURFACE SPECIMEN", color=ACCENT_BLUE, fontsize=11, fontweight="bold")
    ax.text(5, 75.5, "Specimen ID: NEU_Cr_247.bmp  |  Resolution: 200×200 8-bit Grayscale", color=TEXT_MUTED, fontsize=9)

    # Embed sample image
    sample_np = load_sample_img("data/raw/NEU-CLS/Cr_247.bmp")
    # Subplot for image
    img_ax = fig.add_axes([0.05, 0.22, 0.42, 0.48])
    img_ax.imshow(sample_np, cmap="gray")
    img_ax.axis("off")
    # Border
    img_border = patches.Rectangle((0, 0), 1, 1, transform=img_ax.transAxes, fill=False, edgecolor=ACCENT_GREEN, linewidth=2.5)
    img_ax.add_patch(img_border)

    # Significance slider mockup
    ax.text(5, 16, "Conformal Significance Level (α): 0.05 (Confidence 95.0%)", color=TEXT_COLOR, fontsize=9.5)
    s_bg = patches.Rectangle((5, 12), 42, 2, facecolor=CARD_BG, edgecolor=BORDER_COLOR)
    s_fg = patches.Rectangle((5, 12), 8.4, 2, facecolor=ACCENT_BLUE)
    ax.add_patch(s_bg)
    ax.add_patch(s_fg)
    ax.text(13.4, 10, "α = 0.05", color=ACCENT_BLUE, fontsize=8.5, ha="center")

    # Right Panel: Diagnostic Verdict
    p_right = patches.Rectangle((51, 6), 46, 75.5, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_right)
    ax.text(53, 78, "CONFORMAL DIAGNOSTIC REPORT", color=ACCENT_BLUE, fontsize=11, fontweight="bold")

    # Verdict Card (Green)
    v_card = patches.Rectangle((53, 62), 42, 13, facecolor="#064E3B", edgecolor=ACCENT_GREEN, linewidth=2)
    ax.add_patch(v_card)
    ax.text(55, 71, "VERDICT: KNOWN DEFECT IDENTIFIED", color="#34D399", fontsize=13, fontweight="bold")
    ax.text(55, 67, "Conformal Prediction Set: { crazing }   [SINGLETON]", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    ax.text(55, 64, "Closest Class: crazing (Cr) | Distance: 1248.6 | p-value: 0.7842", color="#A7F3D0", fontsize=9.5)

    # Mahalanobis Distance Metrics Bar Chart
    ax.text(53, 57, "Mahalanobis Distances (Inflated Sigma Metric):", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    classes = ["crazing (Cr)", "inclusion (In)", "patches (Pa)", "pitted (PS)", "rolled-in (RS)"]
    dists = [1248.6, 6842.1, 7921.4, 9110.5, 6230.8]
    p_vals = [0.784, 0.000, 0.000, 0.000, 0.001]
    
    for idx, (c_name, d_val, pv) in enumerate(zip(classes, dists, p_vals)):
        cy = 50 - idx * 7.5
        c_bg = CARD_BG
        ax.add_patch(patches.Rectangle((53, cy), 42, 6, facecolor=c_bg, edgecolor=BORDER_COLOR))
        
        # Distance bar
        max_d = 10000.0
        bar_len = min(22, (d_val / max_d) * 22)
        bar_color = ACCENT_GREEN if idx == 0 else ACCENT_BLUE
        ax.add_patch(patches.Rectangle((69, cy + 1.2), bar_len, 3.6, facecolor=bar_color))
        
        ax.text(54.5, cy + 3, c_name, color=TEXT_COLOR, fontsize=9, va="center")
        ax.text(93, cy + 3, f"d={d_val:.1f} (p={pv:.3f})", color=TEXT_MUTED, fontsize=8.5, ha="right", va="center")

    # Routing status
    ax.text(53, 9, "Routing Decision: ROUTE TO CR-SORTING BUCKET  |  No Operator Escalation Needed", color=ACCENT_GREEN, fontsize=9.5, fontweight="bold")

    fig.savefig(OUT_DIR / "01_inspector_known.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_02_inspector_unknown():
    fig, ax = create_base_canvas(
        "Real-Time Surface Defect Inspector",
        "Novel defect detection alert: Out-of-Distribution specimen triggers escalation",
        0
    )

    # Left Panel: Image Viewer
    p_left = patches.Rectangle((3, 6), 46, 75.5, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_left)
    ax.text(5, 78, "INPUT SURFACE SPECIMEN", color=ACCENT_BLUE, fontsize=11, fontweight="bold")
    ax.text(5, 75.5, "Specimen ID: NEU_Sc_241.bmp  |  Surface: Scratches (Held-Out Unknown)", color=TEXT_MUTED, fontsize=9)

    # Embed sample image
    sample_np = load_sample_img("data/raw/NEU-CLS/Sc_241.bmp")
    img_ax = fig.add_axes([0.05, 0.22, 0.42, 0.48])
    img_ax.imshow(sample_np, cmap="gray")
    img_ax.axis("off")
    # Alert Red Border
    img_border = patches.Rectangle((0, 0), 1, 1, transform=img_ax.transAxes, fill=False, edgecolor=ACCENT_RED, linewidth=3.5)
    img_ax.add_patch(img_border)

    ax.text(5, 16, "Conformal Significance Level (α): 0.05 (Confidence 95.0%)", color=TEXT_COLOR, fontsize=9.5)
    s_bg = patches.Rectangle((5, 12), 42, 2, facecolor=CARD_BG, edgecolor=BORDER_COLOR)
    s_fg = patches.Rectangle((5, 12), 8.4, 2, facecolor=ACCENT_BLUE)
    ax.add_patch(s_bg)
    ax.add_patch(s_fg)

    # Right Panel: Diagnostic Verdict
    p_right = patches.Rectangle((51, 6), 46, 75.5, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_right)
    ax.text(53, 78, "CONFORMAL DIAGNOSTIC REPORT", color=ACCENT_BLUE, fontsize=11, fontweight="bold")

    # Verdict Card (Red Alert)
    v_card = patches.Rectangle((53, 61), 42, 14, facecolor="#450A0A", edgecolor=ACCENT_RED, linewidth=2.5)
    ax.add_patch(v_card)
    ax.text(55, 71.5, "VERDICT: FLAGGED — NOVEL DEFECT DETECTED", color="#FCA5A5", fontsize=13, fontweight="bold")
    ax.text(55, 67, "Conformal Prediction Set: { }  [EMPTY SET: REJECT ALL]", color=ACCENT_RED, fontsize=11, fontweight="bold")
    ax.text(55, 63.5, "Min Mahalanobis d = 3380.2 | All p-values < 0.05 | Out-of-Distribution", color="#FEE2E2", fontsize=9.5)

    # Mahalanobis Distance Metrics Bar Chart
    ax.text(53, 56.5, "Mahalanobis Distances to Active Baseline Classes:", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    classes = ["crazing (Cr)", "inclusion (In)", "patches (Pa)", "pitted (PS)", "rolled-in (RS)"]
    dists = [3380.2, 5920.4, 8214.7, 7490.1, 6140.8]
    p_vals = [0.008, 0.000, 0.000, 0.000, 0.000]
    
    for idx, (c_name, d_val, pv) in enumerate(zip(classes, dists, p_vals)):
        cy = 49.5 - idx * 7.5
        ax.add_patch(patches.Rectangle((53, cy), 42, 6, facecolor=CARD_BG, edgecolor=BORDER_COLOR))
        
        max_d = 10000.0
        bar_len = min(22, (d_val / max_d) * 22)
        ax.add_patch(patches.Rectangle((69, cy + 1.2), bar_len, 3.6, facecolor=ACCENT_RED))
        
        ax.text(54.5, cy + 3, c_name, color=TEXT_COLOR, fontsize=9, va="center")
        ax.text(93, cy + 3, f"d={d_val:.1f} (p={pv:.3f} < 0.05)", color="#FCA5A5", fontsize=8.5, ha="right", va="center")

    # Routing Alert
    ax.text(53, 9, "CRITICAL ACTION: Automatically Logged to Operator Review Queue (Priority: HIGH)", color=ACCENT_RED, fontsize=9.5, fontweight="bold")

    fig.savefig(OUT_DIR / "02_inspector_unknown.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_03_production_stream():
    fig, ax = create_base_canvas(
        "Simulated Production Line Stream",
        "Continuous automated rolling-mill surface scanning at line speed",
        1
    )

    # Top Control Bar
    p_ctrl = patches.Rectangle((3, 71.5), 94, 10, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_ctrl)
    
    # Buttons
    ax.add_patch(patches.Rectangle((5, 73.5), 14, 6, facecolor="#064E3B", edgecolor=ACCENT_GREEN, linewidth=1.5))
    ax.text(12, 76.5, "▶ RUNNING", color="#34D399", fontsize=10, fontweight="bold", ha="center", va="center")
    
    ax.add_patch(patches.Rectangle((20.5, 73.5), 12, 6, facecolor=CARD_BG, edgecolor=BORDER_COLOR))
    ax.text(26.5, 76.5, "❚❚ Pause", color=TEXT_MUTED, fontsize=9.5, ha="center", va="center")

    ax.add_patch(patches.Rectangle((34, 73.5), 12, 6, facecolor=CARD_BG, edgecolor=BORDER_COLOR))
    ax.text(40, 76.5, "■ Stop", color=TEXT_MUTED, fontsize=9.5, ha="center", va="center")

    # Burst Injection Button
    ax.add_patch(patches.Rectangle((48, 73.5), 23, 6, facecolor="#7C2D12", edgecolor=ACCENT_AMBER, linewidth=1.5))
    ax.text(59.5, 76.5, "⚡ Inject 5-Defect Burst (Sc)", color="#FDE68A", fontsize=9.5, fontweight="bold", ha="center", va="center")

    # Line Speed Slider
    ax.text(74, 77.5, "Line Speed: 1.0 fps (60 ppm)", color=TEXT_COLOR, fontsize=9)
    ax.add_patch(patches.Rectangle((74, 74), 21, 2, facecolor=CARD_BG, edgecolor=BORDER_COLOR))
    ax.add_patch(patches.Rectangle((74, 74), 7, 2, facecolor=ACCENT_BLUE))

    # KPI Summary Cards
    kpis = [
        ("Frames Inspected", "184", TEXT_COLOR),
        ("Normal / Known", "158 (85.9%)", ACCENT_GREEN),
        ("Flagged / Anomalies", "26 (14.1%)", ACCENT_RED),
        ("Mean Latency", "39.4 ms", ACCENT_BLUE)
    ]
    for idx, (label, val, col) in enumerate(kpis):
        kx = 3 + idx * 24
        ax.add_patch(patches.Rectangle((kx, 59), 22, 10, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2))
        ax.text(kx + 2, 66.5, label, color=TEXT_MUTED, fontsize=9)
        ax.text(kx + 2, 62, val, color=col, fontsize=13, fontweight="bold")

    # Live Ticker Stream Table
    p_table = patches.Rectangle((3, 6), 94, 50.5, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_table)
    ax.text(5, 53, "LIVE SCANNING FEED (Recent Rolling Inspection Frames)", color=ACCENT_BLUE, fontsize=11, fontweight="bold")

    # Table Header
    headers = ["Time", "Frame ID", "True Defect", "Closest Class", "Distance", "Verdict", "Operator Action"]
    col_x = [5, 17, 30, 44, 57, 70, 84]
    for hx, htitle in zip(col_x, headers):
        ax.text(hx, 49.5, htitle, color=TEXT_MUTED, fontsize=9, fontweight="bold")
    ax.plot([4, 95], [48.2, 48.2], color=BORDER_COLOR, linewidth=1)

    # Rows
    rows = [
        ("14:22:18", "frame_0184", "crazing", "crazing", "1284.2", "PASS", "Normal Routing"),
        ("14:22:17", "frame_0183", "patches", "patches", "1542.0", "PASS", "Normal Routing"),
        ("14:22:16", "frame_0182", "scratches", "crazing", "3410.5", "FLAG", "Escalated to Review"),
        ("14:22:15", "frame_0181", "scratches", "inclusion", "3820.1", "FLAG", "Escalated to Review"),
        ("14:22:14", "frame_0180", "inclusion", "inclusion", "1412.8", "PASS", "Normal Routing"),
        ("14:22:13", "frame_0179", "pitted", "pitted", "1190.4", "PASS", "Normal Routing"),
        ("14:22:12", "frame_0178", "rolled-in", "rolled-in", "1620.3", "PASS", "Normal Routing"),
    ]
    for idx, r in enumerate(rows):
        ry = 44 - idx * 5.2
        is_flag = (r[5] == "FLAG")
        row_bg = "#3B1010" if is_flag else (CARD_BG if idx % 2 == 0 else PANEL_BG)
        ax.add_patch(patches.Rectangle((4, ry - 1.2), 92, 4.6, facecolor=row_bg, edgecolor=ACCENT_RED if is_flag else BORDER_COLOR, linewidth=1 if is_flag else 0.5))
        
        ax.text(col_x[0], ry + 0.8, r[0], color=TEXT_MUTED, fontsize=8.5)
        ax.text(col_x[1], ry + 0.8, r[1], color=TEXT_COLOR, fontsize=8.5, fontweight="bold")
        ax.text(col_x[2], ry + 0.8, r[2], color=TEXT_COLOR, fontsize=8.5)
        ax.text(col_x[3], ry + 0.8, r[3], color=TEXT_COLOR, fontsize=8.5)
        ax.text(col_x[4], ry + 0.8, r[4], color=TEXT_MUTED, fontsize=8.5)
        
        v_col = ACCENT_RED if is_flag else ACCENT_GREEN
        ax.text(col_x[5], ry + 0.8, r[5], color=v_col, fontsize=8.5, fontweight="bold")
        ax.text(col_x[6], ry + 0.8, r[6], color=TEXT_COLOR, fontsize=8.5)

    fig.savefig(OUT_DIR / "03_production_stream.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_04_operator_queue():
    fig, ax = create_base_canvas(
        "Operator Review Queue & Novelty Triage",
        "Active escalation queue for flagged anomaly frames requiring metallurgist review",
        2
    )

    # Left: Backlog Table
    p_left = patches.Rectangle((3, 6), 56, 75.5, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_left)
    ax.text(5, 78, "UNREVIEWED ANOMALY BACKLOG (4 Pending)", color=ACCENT_AMBER, fontsize=11, fontweight="bold")

    headers = ["ID", "Timestamp", "Closest Class", "Min Dist", "Status"]
    col_x = [5, 13, 27, 40, 50]
    for hx, ht in zip(col_x, headers):
        ax.text(hx, 74, ht, color=TEXT_MUTED, fontsize=8.5, fontweight="bold")
    ax.plot([4, 58], [72.5, 72.5], color=BORDER_COLOR, linewidth=1)

    items = [
        ("#104", "14:22:16", "crazing (Cr)", "3410.5", "SELECTED"),
        ("#103", "14:22:15", "inclusion (In)", "3820.1", "QUEUED"),
        ("#098", "14:19:42", "rolled-in (RS)", "4102.8", "QUEUED"),
        ("#091", "14:15:10", "crazing (Cr)", "3395.2", "QUEUED"),
    ]
    for idx, it in enumerate(items):
        iy = 67 - idx * 7
        is_sel = (it[4] == "SELECTED")
        ax.add_patch(patches.Rectangle((4, iy - 1.5), 54, 6, facecolor=CARD_BG if is_sel else PANEL_BG, edgecolor=ACCENT_BLUE if is_sel else BORDER_COLOR, linewidth=1.5 if is_sel else 0.5))
        ax.text(col_x[0], ry:=iy+1.2, it[0], color=ACCENT_BLUE if is_sel else TEXT_COLOR, fontsize=8.5, fontweight="bold")
        ax.text(col_x[1], ry, it[1], color=TEXT_MUTED, fontsize=8.5)
        ax.text(col_x[2], ry, it[2], color=TEXT_COLOR, fontsize=8.5)
        ax.text(col_x[3], ry, it[3], color=TEXT_MUTED, fontsize=8.5)
        ax.text(col_x[4], ry, it[4], color=ACCENT_BLUE if is_sel else TEXT_MUTED, fontsize=8, fontweight="bold")

    # Staging Bucket Section
    p_stage = patches.Rectangle((4, 8), 54, 28, facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1)
    ax.add_patch(p_stage)
    ax.text(6, 32, "5-SHOT ENROLLMENT STAGING BUCKET (Target: 5 Exemplars)", color=ACCENT_BLUE, fontsize=10, fontweight="bold")
    ax.text(6, 29.5, "Accumulated: 3 / 5 exemplars for pending novel class candidate", color=TEXT_MUTED, fontsize=8.5)

    for b_idx in range(5):
        bx = 6 + b_idx * 10.4
        is_filled = (b_idx < 3)
        b_box = patches.Rectangle((bx, 13), 9.2, 14, facecolor="#0F172A" if is_filled else PANEL_BG, edgecolor=ACCENT_GREEN if is_filled else BORDER_COLOR, linewidth=1.5 if is_filled else 1)
        ax.add_patch(b_box)
        if is_filled:
            ax.text(bx + 4.6, 21, f"Frame\n#0{98 + b_idx*3}", color=TEXT_COLOR, fontsize=8, ha="center", va="center")
            ax.text(bx + 4.6, 15, "STAGED", color=ACCENT_GREEN, fontsize=7.5, fontweight="bold", ha="center")
        else:
            ax.text(bx + 4.6, 20, "[Slot Empty]", color=TEXT_MUTED, fontsize=7.5, ha="center")

    # Right: Detailed Review Card
    p_right = patches.Rectangle((61, 6), 36, 75.5, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_right)
    ax.text(63, 78, "REVIEW SPECIMEN: #104", color=ACCENT_BLUE, fontsize=11, fontweight="bold")

    # Thumbnail
    sample_np = load_sample_img("data/raw/NEU-CLS/Sc_241.bmp")
    img_ax = fig.add_axes([0.63, 0.52, 0.32, 0.22])
    img_ax.imshow(sample_np, cmap="gray")
    img_ax.axis("off")

    # Metadata
    ax.text(63, 47, "Anomaly Score: d = 3410.5 (Critical Novelty)", color=ACCENT_RED, fontsize=9.5, fontweight="bold")
    ax.text(63, 44, "Closest Known Base Class: crazing (p = 0.006)", color=TEXT_MUTED, fontsize=8.5)
    ax.text(63, 41, "Visual Features: Longitudinal linear scouring", color=TEXT_COLOR, fontsize=8.5)

    # Actions
    ax.text(63, 35, "Operator Triage Actions:", color=TEXT_COLOR, fontsize=9.5, fontweight="bold")

    # Action 1
    ax.add_patch(patches.Rectangle((63, 27), 32, 6, facecolor=CARD_BG, edgecolor=BORDER_COLOR))
    ax.text(65, 30, "✔ Confirm as Known Class (Override)", color=TEXT_COLOR, fontsize=8.5, va="center")

    # Action 2: Stage
    ax.add_patch(patches.Rectangle((63, 19), 32, 6, facecolor="#1E3A8A", edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(65, 22, "➕ Stage Frame for 5-Shot Enrollment", color="#93C5FD", fontsize=8.5, fontweight="bold", va="center")

    # Action 3: Dismiss
    ax.add_patch(patches.Rectangle((63, 11), 32, 6, facecolor=CARD_BG, edgecolor=BORDER_COLOR))
    ax.text(65, 14, "✖ Dismiss as Image Noise / Sensor Artifact", color=TEXT_MUTED, fontsize=8.5, va="center")

    fig.savefig(OUT_DIR / "04_operator_queue.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_05_enrollment_gate():
    fig, ax = create_base_canvas(
        "Novel Defect Enrollment Safety Gate",
        "Deterministic pre-flight verification: Novelty, Consistency, and Overlap rules",
        3
    )

    # Left Panel: Candidate Specs & Exemplars
    p_left = patches.Rectangle((3, 6), 46, 75.5, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_left)
    ax.text(5, 78, "CANDIDATE DEFECT CLASS SPECIFICATION", color=ACCENT_BLUE, fontsize=11, fontweight="bold")

    ax.text(5, 73.5, "Candidate Class Name:", color=TEXT_MUTED, fontsize=9)
    ax.add_patch(patches.Rectangle((5, 68), 42, 4.5, facecolor=CARD_BG, edgecolor=ACCENT_BLUE, linewidth=1.2))
    ax.text(7, 70.2, "Sc_scratches", color=TEXT_COLOR, fontsize=10, fontweight="bold", va="center")

    ax.text(5, 64, "5-Shot Exemplar Images (from Staging Queue):", color=TEXT_MUTED, fontsize=9)

    for i in range(5):
        thumb_ax = fig.add_axes([0.05 + i * 0.082, 0.44, 0.075, 0.12])
        sample_np = load_sample_img("data/raw/NEU-CLS/Sc_241.bmp")
        thumb_ax.imshow(sample_np, cmap="gray")
        thumb_ax.axis("off")
        thumb_ax.text(0.5, -0.2, f"Img {i+1}", transform=thumb_ax.transAxes, color=TEXT_MUTED, fontsize=7.5, ha="center")

    # Pre-Flight Rules Report
    p_rules = patches.Rectangle((5, 10), 42, 30, facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1)
    ax.add_patch(p_rules)
    ax.text(7, 36.5, "SAFETY GATE PRE-FLIGHT EVALUATION", color=ACCENT_BLUE, fontsize=10, fontweight="bold")

    checks = [
        ("1. Novelty Check", "PASSED", "Sep ratio: 4.82 > 2.5 min", ACCENT_GREEN),
        ("2. Consistency Check", "PASSED", "Variance: 0.41 < 1.2 max", ACCENT_GREEN),
        ("3. Overlap Check", "WARNING", "Margin to In: 0.94 < 1.0", ACCENT_AMBER),
    ]
    for idx, (c_name, status, detail, col) in enumerate(checks):
        cy = 30 - idx * 7.5
        ax.text(7, cy + 2, c_name, color=TEXT_COLOR, fontsize=9, fontweight="bold")
        ax.text(28, cy + 2, status, color=col, fontsize=9, fontweight="bold")
        ax.text(7, cy - 0.5, detail, color=TEXT_MUTED, fontsize=8)

    # Right Panel: Decision & Operator Override
    p_right = patches.Rectangle((51, 6), 46, 75.5, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_right)
    ax.text(53, 78, "GATE VERDICT & AUDIT JUSTIFICATION", color=ACCENT_BLUE, fontsize=11, fontweight="bold")

    # Gate Verdict Banner (Amber REVIEW)
    ax.add_patch(patches.Rectangle((53, 64), 42, 11, facecolor="#451A03", edgecolor=ACCENT_AMBER, linewidth=2))
    ax.text(55, 71.5, "GATE VERDICT: REVIEW REQUIRED", color="#FBBF24", fontsize=12, fontweight="bold")
    ax.text(55, 67, "Borderline overlap with inclusion class. Enrollment blocked\nunless qualified operator provides verified justification note.", color="#FEF3C7", fontsize=8.5)

    # Justification Input Box
    ax.text(53, 60, "Mandatory Operator Justification Note (>=10 chars):", color=TEXT_COLOR, fontsize=9, fontweight="bold")
    ax.add_patch(patches.Rectangle((53, 44), 42, 14, facecolor=CARD_BG, edgecolor=ACCENT_AMBER, linewidth=1.5))
    note_text = "Verified longitudinal scratch striations distinct from\nsurface inclusion pits. Inspected by Level-3 QA metallurgist\non shift A-2. Approved for active recognition memory."
    ax.text(55, 54, note_text, color=TEXT_COLOR, fontsize=8.5)

    # Action Buttons
    ax.add_patch(patches.Rectangle((53, 28), 42, 10, facecolor="#166534", edgecolor=ACCENT_GREEN, linewidth=2))
    ax.text(74, 33, "✔ COMMIT TO ACTIVE RECOGNITION MEMORY", color="#BBF7D0", fontsize=10.5, fontweight="bold", ha="center", va="center")
    ax.text(74, 29.5, "Appends class to memory log & rebuilds centroid registry", color="#86EFAC", fontsize=8, ha="center")

    ax.add_patch(patches.Rectangle((53, 14), 42, 8, facecolor=CARD_BG, edgecolor=BORDER_COLOR))
    ax.text(74, 18, "✖ Abort & Return to Queue", color=TEXT_MUTED, fontsize=9.5, ha="center", va="center")

    fig.savefig(OUT_DIR / "05_enrollment_gate.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_06_records_statistics():
    fig, ax = create_base_canvas(
        "Production Inspection Analytics & Audit Log",
        "Operational telemetry, defect distribution breakdown, and audit exports",
        4
    )

    # KPI Header Cards
    kpis = [
        ("Total Scanned", "1,248", TEXT_COLOR),
        ("Defect Flags", "168 (13.5%)", ACCENT_RED),
        ("Unreviewed Queue", "4 pending", ACCENT_AMBER),
        ("Enrolled Classes", "1 added (Sc)", ACCENT_GREEN),
    ]
    for idx, (label, val, col) in enumerate(kpis):
        kx = 3 + idx * 24
        ax.add_patch(patches.Rectangle((kx, 67), 22, 14, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2))
        ax.text(kx + 2, 76.5, label, color=TEXT_MUTED, fontsize=9.5)
        ax.text(kx + 2, 71, val, color=col, fontsize=14, fontweight="bold")

    # Left Chart: Verdict Distribution Donut
    donut_ax = fig.add_axes([0.05, 0.22, 0.42, 0.42], facecolor=PANEL_BG)
    sizes = [86.5, 13.5]
    colors = [ACCENT_BLUE, ACCENT_RED]
    wedges, texts = donut_ax.pie(sizes, colors=colors, startangle=90, counterclock=False,
                                 wedgeprops=dict(width=0.45, edgecolor=BG_COLOR, linewidth=2))
    donut_ax.text(0, 0, "86.5%\nNORMAL", ha="center", va="center", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    donut_ax.set_title("Inspection Verdict Ratio (Normal vs Novel Flag)", color=TEXT_COLOR, fontsize=10, pad=10)

    # Right Chart: Defect Class Frequency
    bar_ax = fig.add_axes([0.54, 0.26, 0.41, 0.35], facecolor=PANEL_BG)
    bar_ax.set_facecolor(CARD_BG)
    classes = ["crazing", "inclusion", "patches", "pitted", "rolled-in", "Sc_scratches"]
    counts = [380, 240, 260, 110, 90, 168]
    bar_cols = [ACCENT_BLUE, ACCENT_BLUE, ACCENT_BLUE, ACCENT_BLUE, ACCENT_BLUE, ACCENT_AMBER]
    bars = bar_ax.barh(classes, counts, color=bar_cols, height=0.6)
    bar_ax.set_title("Defect Distribution Across Manufacturing Shift", color=TEXT_COLOR, fontsize=10)
    bar_ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in bar_ax.spines.values():
        spine.set_color(BORDER_COLOR)

    # Bottom Audit Export Bar
    p_bot = patches.Rectangle((3, 6), 94, 9, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_bot)
    ax.text(5, 10.5, "Export Operational Audit Log:", color=TEXT_COLOR, fontsize=10, fontweight="bold")
    
    # Export buttons
    ax.add_patch(patches.Rectangle((36, 7.5), 18, 6, facecolor=CARD_BG, edgecolor=ACCENT_BLUE, linewidth=1.2))
    ax.text(45, 10.5, "📥 Download JSONL", color=ACCENT_BLUE, fontsize=9, fontweight="bold", ha="center", va="center")

    ax.add_patch(patches.Rectangle((56, 7.5), 18, 6, facecolor=CARD_BG, edgecolor=ACCENT_BLUE, linewidth=1.2))
    ax.text(65, 10.5, "📥 Download CSV", color=ACCENT_BLUE, fontsize=9, fontweight="bold", ha="center", va="center")

    fig.savefig(OUT_DIR / "06_records_statistics.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_07_model_evidence():
    fig, ax = create_base_canvas(
        "Model Architecture & Scientific Benchmark Evidence",
        "Empirical performance, confusion matrix, and research protocol status",
        5
    )

    # Top Warning Disclaimer Banner
    p_banner = patches.Rectangle((3, 70), 94, 11.5, facecolor="#451A03", edgecolor=ACCENT_AMBER, linewidth=2)
    ax.add_patch(p_banner)
    ax.text(5, 78, "⚠️ HISTORICAL v1.0 MODEL BENCHMARK — EXPLORATORY BASELINE", color="#FBBF24", fontsize=11, fontweight="bold")
    disclaimer = (
        "Model selection in v1.0 evaluated test-pool data during architecture search (leakage). "
        "Reported metrics are historical baselines for application engineering demonstration. "
        "Strict non-leaking Protocol v1.1 evaluation (nested B1/B2 retuning) is currently pending."
    )
    ax.text(5, 73.5, disclaimer, color="#FEF3C7", fontsize=8.5)

    # Left: Benchmark Metrics Grid
    p_left = patches.Rectangle((3, 6), 46, 61, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_left)
    ax.text(5, 63, "BENCHMARK METRICS (Primary Run H=Sc, s=0, B0_L4)", color=ACCENT_BLUE, fontsize=10.5, fontweight="bold")

    metrics = [
        ("Closed-Set Accuracy", "100.0% (200 / 200)", ACCENT_GREEN),
        ("Novelty Detection AUROC", "0.9829", ACCENT_GREEN),
        ("False Reject Rate (FRR @ 95%)", "1.50%", ACCENT_GREEN),
        ("Novelty Rejection Rate", "88.80%", ACCENT_BLUE),
        ("5-Shot Continual Mean Recall", "12.96% (historical)", ACCENT_AMBER),
        ("Backbone Feature Extractor", "ResNet-18 (L4, 512-d)", TEXT_COLOR),
        ("ONNX Engine Peak Memory", "299.94 MB (<420 MB limit)", ACCENT_GREEN),
    ]
    for idx, (m_label, m_val, m_col) in enumerate(metrics):
        my = 57 - idx * 7.2
        ax.add_patch(patches.Rectangle((5, my - 1.5), 42, 6, facecolor=CARD_BG, edgecolor=BORDER_COLOR))
        ax.text(6.5, my + 1.2, m_label, color=TEXT_COLOR, fontsize=8.5)
        ax.text(45.5, my + 1.2, m_val, color=m_col, fontsize=8.5, fontweight="bold", ha="right")

    # Right: Confusion Matrix
    p_right = patches.Rectangle((51, 6), 46, 61, facecolor=PANEL_BG, edgecolor=BORDER_COLOR, linewidth=1.2)
    ax.add_patch(p_right)
    ax.text(53, 63, "COMMITTED TEST CONFUSION MATRIX (reports/results_*.json)", color=ACCENT_BLUE, fontsize=10.5, fontweight="bold")

    # Heatmap Table
    conf_classes = ["Cr", "In", "Pa", "PS", "RS", "Unk(Sc)"]
    matrix_data = [
        [50,  0,  0,  0,  0,  0],
        [ 0, 50,  0,  0,  0,  0],
        [ 0,  0, 50,  0,  0,  0],
        [ 0,  0,  0, 50,  0,  0],
        [ 0,  0,  0,  0, 50,  0],
        [ 1,  2,  1,  1,  1, 44],
    ]
    
    mat_ax = fig.add_axes([0.55, 0.16, 0.38, 0.38])
    mat_ax.set_facecolor(CARD_BG)
    im = mat_ax.imshow(matrix_data, cmap="Blues")
    mat_ax.set_xticks(range(6))
    mat_ax.set_yticks(range(6))
    mat_ax.set_xticklabels(conf_classes, color=TEXT_COLOR, fontsize=8)
    mat_ax.set_yticklabels(conf_classes, color=TEXT_COLOR, fontsize=8)
    mat_ax.set_xlabel("Predicted Class", color=TEXT_MUTED, fontsize=8.5)
    mat_ax.set_ylabel("True Ground Truth", color=TEXT_MUTED, fontsize=8.5)

    for i in range(6):
        for j in range(6):
            val = matrix_data[i][j]
            col = "white" if val > 20 else TEXT_MUTED
            mat_ax.text(j, i, str(val), ha="center", va="center", color=col, fontsize=8, fontweight="bold")

    ax.text(53, 9, "Source Artifact: reports/results_H=Sc_s=0_B0_L4.json (Frozen commit b75b0b8)", color=TEXT_MUTED, fontsize=8)

    fig.savefig(OUT_DIR / "07_model_evidence.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    print("Generating NovaSight Faculty Review screens...")
    generate_01_inspector_known()
    print("[OK] 01_inspector_known.png")
    generate_02_inspector_unknown()
    print("[OK] 02_inspector_unknown.png")
    generate_03_production_stream()
    print("[OK] 03_production_stream.png")
    generate_04_operator_queue()
    print("[OK] 04_operator_queue.png")
    generate_05_enrollment_gate()
    print("[OK] 05_enrollment_gate.png")
    generate_06_records_statistics()
    print("[OK] 06_records_statistics.png")
    generate_07_model_evidence()
    print("[OK] 07_model_evidence.png")
    print("All 7 screens successfully generated in reports/app_mvp/screens/.")


if __name__ == "__main__":
    main()
