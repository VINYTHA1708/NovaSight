---
title: NovaSight
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
python_version: "3.12"
---

# NovaSight — Industrial Open-Set Steel Surface Defect Inspector

[![Python 3.12](https://img.shields.io/badge/python-3.12.10-blue.svg)](https://www.python.org/downloads/release/python-31210/)
[![Runtime ONNX](https://img.shields.io/badge/runtime-ONNX%20Opset%2017-green.svg)](https://onnxruntime.ai/)
[![UI Gradio](https://img.shields.io/badge/ui-Gradio%206.30.0-orange.svg)](https://www.gradio.app/)
[![RAM Peak](https://img.shields.io/badge/peak%20RAM-308.96%20MB%20(%3C512MB)-brightgreen.svg)](reports/app_mvp/app_memory_profile.json)
[![Tests Passing](https://img.shields.io/badge/tests-78%20passed%2C%201%20skipped-success.svg)](tests/)
[![Team](https://img.shields.io/badge/Team-No.%2021-purple.svg)](#-team-member-details)

**NexAI Hack 2026 (British Council) · Problem PS 2.3 "Unknown-Defect Steel Inspector" · Track 2 Manufacturing QA**

NovaSight is a research-grade, industrial surface anomaly and open-set defect inspection platform engineered for continuous hot-rolled steel strip manufacturing. It resolves the critical failure mode of standard computer vision classifiers—**silent misclassification of novel defect classes**—by pairing closed-set recognition with finite-sample **Mondrian conformal prediction guarantees** and a **deterministic 5-shot safety-gated continual enrollment engine**.

---

## 📌 Problem Statement & Solution Overview

### The Industrial Problem
Modern high-speed steel rolling mills process strip steel at velocities exceeding 20 meters per second. Automated surface inspection systems (ASIS) powered by conventional deep learning classifiers are strictly **closed-set**: they force every visual anomaly into one of several pre-trained defect categories. When an unfamiliar defect emerges (caused by rolling wear, lubrication contamination, or alloy changes), standard classifiers assign it to a known class with false overconfidence. This leads to defective steel coils being shipped to automotive and aerospace clients, causing catastrophic structural failures and millions in recall claims.

### The NovaSight Solution
NovaSight treats industrial defect inspection as a joint **classification, out-of-distribution (OOD) rejection, and few-shot adaptation** challenge:
1. **Statistical Rejection Guarantees**: Computes class-conditional inflated Mahalanobis distance scores in a learned representation space and calibrates non-conformity scores using Mondrian conformal prediction. At significance level $\alpha = 0.05$, known defects are classified with confidence, while unfamiliar defects yield an empty prediction set ($\Gamma^\alpha(x) = \emptyset$), triggering immediate operator review.
2. **Sub-512MB Cloud & Edge Runtime**: Replaces heavy deep learning frameworks (`torch`, `torchvision`, `scikit-learn`) in production with a streamlined **ONNX Runtime** engine and custom linear algebra. Cold-start memory is just **205 MB**, and peak operational RAM is **308 MB**, fitting comfortably inside free-tier cloud instances (Render, Hugging Face Spaces) and edge industrial PCs without OOM risk.
3. **Safety-Gated 5-Shot Continual Enrollment**: Enables plant metallurgists to register newly identified defect classes right from the factory floor using only 5 exemplar images. A deterministic mathematical safety gate verifies novelty separation, intra-class consistency, and known-class overlap, blocking invalid enrollments (`REJECT`) and requiring operator audit notes (`REVIEW`) to prevent memory corruption.

### Mandatory Requirements Matrix (M1 / M2 / M3)

| ID | Hackathon Requirement | System Implementation & Status |
|:---:|:---|:---|
| **M1** | Classifier trained on 5 of 6 NEU-CLS classes; held-out class stated clearly in documentation | ✅ **Held-out class: Scratches (`Sc`)**. Model trained strictly on Crazing (`Cr`), Inclusion (`In`), Patches (`Pa`), Pitted Surface (`PS`), and Rolled-in Scale (`RS`). |
| **M2** | Unknown-detection method flags held-out images as "unknown"; evidenced via confusion matrix | ✅ **Mondrian Conformal + Min Inflated Mahalanobis**. Achieves **0.9829 AUROC** on novelty detection and **88.80% strict unknown rejection** at $\alpha=0.05$. Reported in 6-class confusion matrix. |
| **M3** | Operator UI: upload image $\rightarrow$ defect class or "unknown, send to operator"; label 5 examples $\rightarrow$ register class without full retraining | ✅ **Industrial Gradio Dashboard (`app.py`)**. Includes single-frame inspection, automated production streaming, operator triage queue, 5-shot enrollment gate, audit telemetry, and benchmark evidence. |

---

## 🏗 System Architecture & Technology Stack

NovaSight cleanly decouples offline representation learning and statistical artifact generation from the production inference and persistence runtime:

```mermaid
flowchart TD
    subgraph Offline Research Pipeline
        A["NEU-CLS Dataset (1,800 BMPs)"] --> B["Stratified Split Manifests (S1)"]
        B --> C["ResNet-18 Backbone Feature Extractor"]
        C --> D["Frozen Artifact: B0_L4_H=Sc_s=0.npz"]
        C --> E["ONNX Exporter (Opset 17)"]
        E --> F["models/onnx/resnet18_b0_l4.onnx (44.7 MB)"]
    end

    subgraph Production Runtime Engine (Torch-Free)
        G["Camera Frame (200x200 BMP)"] --> H["src/runtime/preprocess.py (PIL / NumPy)"]
        H --> I["src/runtime/onnx_extractor.py (ONNX Runtime)"]
        F --> I
        I --> J["512-dim Normalized Embedding"]
        D --> K["src/runtime/scoring.py (Inflated Mahalanobis)"]
        J --> K
        K --> L["Conformal Engine (math.erf Analytic CDF)"]
        L --> M{"Conformal Set Verdict"}
        M -->|"Singleton {c}"| N["Known Defect: Class c"]
        M -->|"Empty Set ∅ / Ambiguous"| O["Flagged: Route to Operator Queue"]
    end

    subgraph Storage & Adaptation Layer
        P["src/app/storage.py (SQLite / Neon Postgres)"] <--> Q["RuntimeMemory Centroid Registry"]
        Q <--> K
        O --> P
        R["src/runtime/gate.py (Novelty / Consistency / Overlap)"] --> Q
    end

    subgraph 6-Tab Industrial Workstation (app.py)
        S["Tab 1: Real-Time Inspector"]
        T["Tab 2: Production Line Stream"]
        U["Tab 3: Operator Triage Queue"]
        V["Tab 4: Enroll New Defect (5-Shot Gate)"]
        W["Tab 5: Records & Telemetry"]
        X["Tab 6: Model & Evidence Benchmark"]
    end

    N --> S
    O --> S
    P --> U
    R --> V
    P --> W
```

### Technology Stack

| Component | Technology | Version / Specification | Rationale |
|---|---|---|---|
| **Language & Core** | Python | `3.12.10` | Fast startup, robust typing, modern runtime |
| **Inference Engine** | ONNX Runtime | `1.17.0+` (CPU) | High-speed, deterministic inference without heavy PyTorch dependencies |
| **Web Interface** | Gradio | `6.30.0` | Reactive industrial dashboard with timer streaming and modal workflows |
| **Image Processing** | Pillow & NumPy | `Pillow >= 10.0`, `NumPy >= 1.26` | Strict 8-bit grayscale verification, bilinear resize, ImageNet standardization |
| **Persistence Layer** | SQLite & PostgreSQL | `sqlite3` (local) / `psycopg3` (Neon cloud) | Zero-configuration local dev with enterprise cloud readiness |
| **Statistical Engine** | Custom math / SciPy | `math.erf` / `scipy.stats` | Zero-overhead normal CDF calculation matching SciPy float precision |
| **System Profiling** | psutil | `psutil >= 5.9` | Real-time process resident set size (RSS) telemetry |

---

## 🚀 Setup and Execution Instructions

### Prerequisites
- Python `3.12` installed (Windows, macOS, or Linux).
- Git installed.

### 1. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/VINYTHA1708/NovaSight.git
cd NovaSight

# Switch to the application MVP branch
git switch app-mvp

# Create and activate a Python 3.12 virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install production application dependencies (Torch-Free)
pip install -r requirements-app.txt
```

> **Note**: For offline research experimentation, model fine-tuning, or running the full legacy evaluation pipeline, use `pip install -r requirements.txt`. For production serving and UI execution, `requirements-app.txt` is minimal, sub-512MB compliant, and installs in under 30 seconds.

### 2. Launch the Application

```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:7860
```

### 3. Running Unit and Integration Tests

The test suite validates data splits, ONNX equivalence, torch-free subprocess isolation, storage CRUD, streaming queues, and safety gate logic:

```bash
# Run all unit, integration, and safety tests
pytest -v

# Run runtime and memory verification only
pytest tests/test_runtime.py tests/test_storage.py tests/test_stream.py -v
```
*Current test suite status: **78 passed, 1 skipped** (PostgreSQL live test cleanly skipped when `DATABASE_URL` is unset).*

### 4. Memory Footprint Measurement

To re-verify that the application complies with the sub-512MB RAM constraint in an isolated clean subprocess:

```bash
python scripts/measure_app.py --out reports/app_mvp/app_memory_profile.json
```

---

## 🌐 Deployment & Local-Demo Details

### Live Deployment Targets
- **Hugging Face Spaces**: Configured via the YAML metadata at the top of this `README.md` (`sdk: gradio`, `sdk_version: 6.30.0`, `python_version: 3.12`).
- **Render.com Web Service**: Production blueprint defined in `render.yaml` with zero-downtime health checking:
  ```yaml
  services:
    - type: web
      name: novasight
      env: python
      plan: free
      buildCommand: pip install -r requirements-app.txt
      startCommand: python app.py
      envVars:
        - key: PYTHON_VERSION
          value: 3.12.10
        - key: DATABASE_URL
          sync: false # Set to your Neon Postgres connection string
  ```

### Dual-Backend Persistence
- **Local Development**: Defaults to local SQLite file at `state/novasight.db`. No setup required.
- **Production Cloud (Neon / Supabase)**: Provide a standard PostgreSQL connection string in the environment:
  ```bash
  export DATABASE_URL="postgresql://user:password@ep-project.neon.tech/neondb?sslmode=require"
  python app.py
  ```
  NovaSight automatically runs idempotent migrations on startup (`classes`, `events`, `inspections` tables) and indicates the active database provider in the UI header.

### Sub-512MB RAM Audit Summary

Empirical memory measurements recorded in `reports/app_mvp/app_memory_profile.json`:

| Lifecycle Stage | Resident Set Size (RSS) | Headroom under 512 MB | Status |
|:---|:---:|:---:|:---:|
| **Clean App Import** | 205.56 MB | 306.44 MB | ✅ Pass |
| **ONNX Session Load** | 205.56 MB | 306.44 MB | ✅ Pass |
| **20 Streamed Inspections** | 291.96 MB | 220.04 MB | ✅ Pass |
| **5-Shot Gate & Class Commit** | 297.55 MB | 214.45 MB | ✅ Pass |
| **Peak Memory Observed** | **308.96 MB** | **203.04 MB (39.7%)** | ✅ **Pass** ($\le 420.0$ MB) |

---

## 🖥 6-Tab Industrial Workstation Tour

NovaSight provides a tailored industrial UI designed for line operators, metallurgical engineers, and quality assurance leads:

| Tab | Name | Operational Purpose |
|:---:|:---|:---|
| **Tab 1** | **Real-Time Inspector** | Single-image diagnostic workstation. Displays specimen preview, adjustable conformal significance slider ($\alpha$), verdict badge (Known / Flagged Novel), and Mahalanobis distance bar charts. |
| **Tab 2** | **Production Line** | Rolling mill streaming simulation. Configurable line speed (0.5–5.0 fps), start/pause/stop controls, live defect burst injection, and real-time scrolling inspection feed. |
| **Tab 3** | **Operator Queue** | Unreviewed anomaly backlog. Displays flagged frames for metallurgical triage. Operators can confirm known defect overrides, dismiss camera noise, or stage specimens into the 5-shot enrollment bucket. |
| **Tab 4** | **Enroll New Defect** | Pre-flight safety-gated 5-shot enrollment. Evaluates novelty separation ($D_{\text{sep}} \ge 2.5$), intra-class consistency ($\text{Var} \le 1.2$), and class overlap margins. Enforces mandatory justification notes on `REVIEW` warnings. |
| **Tab 5** | **Records & Telemetry** | Shift telemetry and compliance audits. Renders KPI summary cards (total scanned, flag rate, mean latency), defect distribution charts, recent inspection audit log, and one-click JSONL/CSV downloads. |
| **Tab 6** | **Model & Evidence** | Scientific transparency portal. Presents the committed test confusion matrix, model card parameters, and historical benchmark disclaimers. |

*Visual screenshot captures and architectural diagrams are available in [FACULTY_REVIEW_0_APP.md](reports/app_mvp/FACULTY_REVIEW_0_APP.md).*

---

## 📊 Benchmark Results & Scientific Transparency

### Primary Run Benchmark (B0/L4, Sc Held-Out, Seed 0)

> **⚠️ Scientific Status Disclosure**: The metrics below represent the **historical Protocol v1.0 baseline** evaluated on the frozen v1 test pool (200 known test images, 125 scratches images) at significance level $\alpha = 0.05$. In protocol v1.0, hyperparameter selection evaluated test pool data during architecture ranking (data leakage). Therefore, these metrics serve as historical engineering baselines. Protocol v1.1 implements strict, non-leaking nested fine-tuning and cross-fitting (implemented on branch `protocol-v1.1` for Gate 4 GPU execution).

| Metric | Measured Value | Numerator / Denominator | Definition / Operational Role |
|:---|:---:|:---:|:---|
| **Closed-Set Accuracy** | **100.0%** | 200 / 200 | Diagnostic accuracy over known base classes |
| **Novelty Detection AUROC** | **0.9829** | — | Area under ROC curve using $N(x) = \min_c \tilde{d}_c(x)$ |
| **FPR @ 95% TPR** | **0.0750** | — | False positive rate at 95% true positive rate |
| **Known Rejection Rate (FRR)** | **0.0150** | 3 / 200 | Strict known false rejection rate ($\Gamma^\alpha(x) = \emptyset$) |
| **Unknown Rejection Rate** | **0.8880** | 111 / 125 | Strict unknown detection rate ($\Gamma^\alpha(x) = \emptyset$) |
| **Known Operator Routing Rate** | **0.8750** | 175 / 200 | Non-singleton routing rate (3 empty + 172 ambiguous) |
| **Unknown Operator Routing Rate**| **0.9520** | 119 / 125 | Non-singleton routing rate (111 empty + 8 ambiguous) |
| **5-Shot Mean Recall (H1)** | **12.96%** | 10 draws | Historical 5-shot continual learning recall on `new_test` |

### Confusion Matrix with Unknown Column (M2 Requirement)

Evaluated at $\alpha = 0.05$ across 200 known test images (40 per class) and 125 held-out unknown scratch images:

| Ground Truth \ Predicted | crazing | inclusion | patches | pitted_surface | rolled-in_scale | **UNKNOWN (Routed)** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **crazing (Cr)** | 0 | 0 | 0 | 0 | 0 | **40** |
| **inclusion (In)** | 0 | 0 | 1 | 0 | 0 | **39** |
| **patches (Pa)** | 0 | 0 | 23 | 0 | 0 | **17** |
| **pitted_surface (PS)** | 0 | 0 | 1 | 0 | 0 | **39** |
| **rolled-in_scale (RS)** | 0 | 0 | 0 | 0 | 0 | **40** |
| **unknown (scratches - Sc)** | 0 | 0 | 6 | 0 | 0 | **119 (95.2%)** |

*Operational Definition*: Any sample yielding an empty prediction set ($\Gamma^\alpha(x) = \emptyset$) or an ambiguous set ($|\Gamma^\alpha(x)| \ge 2$) is safely routed to the metallurgical operator. **119 out of 125 unknown scratches** are successfully intercepted.

---

## 💾 Dataset Attribution and Licence

### Northeastern University (NEU) Surface Defect Database
- **Source**: [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html), Northeastern University, Shenyang, China.
- **Reference**: K. Song and Y. Yan, *"A noise robust method based on completed local binary patterns for hot-rolled steel surface defect detection,"* Applied Surface Science, vol. 285, pp. 858–864, 2013.
- **Dataset Properties**: 1,800 grayscale BMP images ($200 \times 200$ pixels), distributed equally across 6 classes (300 images each):
  1. Crazing (`Cr`)
  2. Inclusion (`In`)
  3. Patches (`Pa`)
  4. Pitted Surface (`PS`)
  5. Rolled-in Scale (`RS`)
  6. Scratches (`Sc`) — *Designated Held-Out Unknown Class for Primary Protocol*
- **Terms & Attribution**: Provided for academic and research use with mandatory citation of the original authors.
- **Dataset Identity Property**: The raw dataset contains one duplicate image pair: `Pa_101.bmp` $\equiv$ `Pa_105.bmp` (identical SHA-256 digest). This is an intrinsic property of the published dataset and is documented in `reports/sanity_scan.json` and `DEV_LOG.md`.

---

## 👥 Team Member Details

**NexAI Hack 2026 (British Council) — Team No: 21**

| Name | Role & Core Contributions |
|:---|:---|
| **VINYTHA S V** | Project Architecture, ONNX Runtime Pipeline, Conformal Calibration & Memory Replay Engine |
| **AKSHAYAA S A** | Manufacturing Stream Simulation, Queue Triage Logic & Dual-Backend Storage Architecture |
| **THEERTHA E** | 5-Shot Enrollment Safety Gate, Quality Pre-Flight Checks & Operational Telemetry |
| **SWASTHIKA P** | Industrial UI Design, Conformal Visual Reporting & Performance Profiling |

---

## 📑 Project Logs & Documentation Links

- 📘 [FACULTY_REVIEW_0_APP.md](reports/app_mvp/FACULTY_REVIEW_0_APP.md) — Comprehensive technical report for Faculty Review 0 (Architecture, RAM proof, ONNX equivalence, Safety rules).
- 📜 [STAGE_LOG.md](reports/app_mvp/STAGE_LOG.md) — Chronological execution audit of application track stages S0 through S10.
- 📓 [DEV_LOG.md](DEV_LOG.md) — Immutable research experiment log, timestamped records, and pre-registered approved policies.
- 🔬 [RESEARCH_SPEC_v1.1.md](docs/RESEARCH_SPEC_v1.1.md) — Complete experimental research specification and protocol guidelines.
- ⚖️ [PROTOCOL_DEVIATIONS.md](docs/PROTOCOL_DEVIATIONS.md) — Transparent disclosure of historical protocol v1.0 deviations and corrective actions.
