# RupeeScan 🔍🇮🇳

[![Python](https://img.shields.io/badge/Python-3.11%20|%203.12%20|%203.13%20|%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%200.143-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20ResNet--18-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV%20Headless-5C3EE8.svg)](https://opencv.org/)
[![Accessibility](https://img.shields.io/badge/Accessibility-WCAG%202.1%20AA-success.svg)](#accessibility-features)
[![Tests](https://img.shields.io/badge/Tests-20%20Passed-brightgreen.svg)](#tests)

> **Empowering Financial Independence for the Visually Impaired through Real-Time Computer Vision.**

RupeeScan is an accessibility-first Indian banknote denomination assistant. Powered by a hybrid PyTorch ResNet-18 and OpenCV vision pipeline, it provides sub-10ms real-time banknote identification via live webcam streaming, image uploads, natural voice announcements, and WCAG-compliant high-contrast controls.

---

## 🌟 About Us & Project Mission

### Why RupeeScan?
In India, while modern banknotes feature tactile bleeding lines and variable sizes, individuals with visual impairments, partial blindness, or age-related vision loss frequently encounter challenges in everyday transactions. Variations in lighting, soiled or folded notes, and rapid merchant exchanges create barriers to financial autonomy.

**RupeeScan was built to solve this challenge.** By bridging advanced deep learning with a universally accessible browser frontend, RupeeScan turns any smartphone, laptop, or webcam-equipped device into an instant, spoken banknote recognizer that works locally with zero cloud latency.

### Core Pillars
- **👁️ Accessibility-First Design:** Engineered from the ground up for screen-reader users, keyboard-only navigators, and low-vision individuals. Features an instant **High-Contrast Theme (Black & Yellow)**, custom focus states, and tactile keyboard shortcuts.
- **🗣️ Natural Voice Synthesis:** Integrates browser-native Web Speech API to immediately announce detected banknotes (e.g. *"Detected Five Hundred Rupees"*), with repeat-on-tap and background throttle prevention.
- **⚡ High-Performance Pipeline:** Optimized for low latency (~7.8 ms per inference, over **120 FPS** on CPU), ensuring fluid video scanning without frame buffering or thermal lag.
- **🛡️ Hybrid Vision Architecture:** Combines deep feature representations from a fine-tuned ResNet-18 with classic computer vision verification (HSV color histogram comparison, Laplacian texture sharpness, and Canny contour bounding-box localization).
- **🔒 Privacy by Design:** All frame processing occurs on the local FastAPI backend server; no user camera feeds or personal data are stored or shared.

### Meet the Team & Developer
- **Lead Developer & Creator:** **Yash Sonawane** ([@yash12991](https://github.com/yash12991))
- **Mission:** Building open, ethical, and high-impact AI accessibility solutions for real-world independence.
- **Contributions & Feedback:** Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yash12991/RupeeScan-AI/issues).

---

## 📸 Key Features

| Feature | Description |
|---|---|
| **Live Webcam Stream** | Real-time WebSocket connection (`/ws/detect`) running at up to 120+ FPS |
| **Static Photo Upload** | Multi-part image upload REST endpoint (`POST /api/detect`) with client-side preview |
| **Voice Synthesis** | Spoken denomination announcements with debounced repeat handling |
| **High Contrast Mode** | Instant toggle between glassmorphic dark theme and high-contrast yellow-on-black |
| **Banknote Localization** | Real-time contour bounding-box tracking overlaid on the video feed |
| **Supported Notes** | **₹10, ₹20, ₹50, ₹100, ₹200, ₹500, and ₹2000** banknotes |
| **Accessibility Shortcuts** | <kbd>Alt + C</kbd> Contrast, <kbd>Alt + V</kbd> Voice, <kbd>Alt + A</kbd> About Us, <kbd>Space</kbd> Scan |

---

## 🏗️ System Architecture

```
[ Web Camera / Image File ]
            │
            ▼ (Base64 Video Frames via WebSocket / Multipart REST)
[ FastAPI Backend (backend/main.py) ]
            │
            ├──► [ Preprocessing & Validation ] (Dimension, byte size, format checks)
            │
            ├──► [ Stage 1: ResNet-18 PyTorch Classifier ] (7 Banknote Denominations)
            │
            ├──► [ Stage 2: OpenCV Guardrails ]
            │         ├─ HSV Histogram Color Correlation
            │         ├─ Laplacian Variance & Canny Texture Checks
            │         └─ Contour Aspect Ratio Bounding-Box Localization
            │
            ▼ (JSON Prediction Payload + Bounding Box Coordinates)
[ Accessible Browser Frontend (Vanilla HTML5 / Modern CSS / JS) ]
            ├─ Dynamic Bounding Box Overlay
            ├─ High-Contrast Visual Indicator
            └─ Spoken Audio via Web Speech API
```

---

## ⚠️ Safety and Project Status

- **Denomination detection is the primary feature.** Authenticity classification is experimental, disabled by default, and must not be treated as proof that a banknote is genuine or counterfeit. Confirm questionable notes through an authorized bank or other official process.
- The YOLO scripts under `training/` are an offline research path for synthetic banknote localization. YOLO is not used by the deployed backend; the live application uses ResNet/classic-CV denomination classification and contour-based localization.
- Detailed model limitations and deployment policy are recorded in [MODEL_CARD.md](MODEL_CARD.md).

---

## 💻 Requirements

- **Operating System:** macOS, Linux, or Windows
- **Python:** 3.11 through 3.14 (fully verified on Python 3.12 and 3.14)
- A modern web browser with camera permissions (Chrome, Edge, Safari, Firefox)
- Pre-trained model weights under `backend/models/`

---

## 🚀 Quickstart & Installation

### 🍎 macOS / 🐧 Linux

```bash
# 1. Clone the repository
git clone https://github.com/yash12991/RupeeScan-AI.git
cd RupeeScan-AI

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt

# 4. Verify system readiness with the built-in diagnostic doctor
python scripts/doctor.py

# 5. Launch the application
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 🪟 Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/yash12991/RupeeScan-AI.git
cd RupeeScan-AI

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

# 4. Run diagnostic doctor
python scripts\doctor.py

# 5. Launch the application
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open your browser at **`http://127.0.0.1:8000`**. Camera APIs function automatically on `localhost`.

---

## ⚙️ Configuration

RupeeScan can be configured via environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `RUPEESCAN_MODEL_DIR` | Directory containing model weights | `backend/models` |
| `RUPEESCAN_FRONTEND_DIR` | Static frontend directory | `frontend` |
| `RUPEESCAN_MAX_IMAGE_BYTES` | REST/WebSocket image payload limit | `10485760` (10 MB) |
| `RUPEESCAN_MAX_IMAGE_PIXELS` | Maximum decoded image pixels | `16000000` (16 MP) |
| `RUPEESCAN_ENABLE_EXPERIMENTAL_AUTHENTICITY` | Opt in to experimental authenticity inference | `false` |
| `RUPEESCAN_DENOMINATION_DATASET` | Real denomination training dataset | `datasets/denomination` |
| `RUPEESCAN_AUTHENTICITY_DATASET` | Independent real/counterfeit research dataset | `datasets/authenticity` |

---

## 🧪 Tests & Diagnostics

### Run Unit and Integration Tests
```bash
pip install -r requirements-dev.txt
pytest
```

The 20-test test suite verifies:
- API health and restrictive browser security headers (`X-Frame-Options`, `Content-Security-Policy`, etc.)
- Upload payload size and pixel dimension limits
- YOLO synthetic dataset split balance and labeling
- Diagnostic doctor contract and model artifact presence
- Frontend accessibility contracts (WCAG attributes, ARIA live-regions, unique element IDs)
- Shipped banknote smoke images inference verification

### Environment Doctor
```bash
python scripts/doctor.py
# Machine-readable output:
python scripts/doctor.py --json
```

### Inference Latency Benchmark
Measure CPU/GPU inference throughput on your system:
```bash
python scripts/benchmark_inference.py --iterations 30
```

---

## 🌐 API Reference

- `GET /api/live` — Lightweight liveness check for container and orchestration probes.
- `GET /api/health` — Comprehensive system diagnostic: model readiness, active engine, device, input limits, and supported classes.
- `POST /api/detect` — Single-image multipart upload endpoint returning denomination, confidence, and bounding box.
- `WS /ws/detect` — Real-time base64 WebSocket video frame processing stream.

---

## 🐳 Docker Deployment

RupeeScan includes a secure, non-root multi-stage Docker configuration:

```bash
# Build Docker image
docker build -t rupeescan:latest .

# Run container on port 8000
docker run --rm -p 8000:8000 rupeescan:latest
```

Access the application at `http://127.0.0.1:8000`.

---

## 📊 Dataset Evaluation & Training

### Real-World Evaluation
Evaluate performance against an independent directory of photographed notes (`real_photos/<denomination>/`):

```bash
python evaluation/evaluate_denomination.py path/to/real_photos --output evaluation/reports/real_photos.json
```

### PyTorch Model Training
```bash
# 1. Prepare deterministic 70/15/15 dataset manifests
python training/prepare_denomination_data.py

# 2. Dry run training validation
python training/train_pytorch.py --dry-run

# 3. Train ResNet-18 model
python training/train_pytorch.py --epochs 20 --batch-size 32 --patience 4
```

---

## 📄 License & Attribution

This project is open-source under the MIT License. Developed and maintained by **[Yash Sonawane](https://github.com/yash12991)**. If you use RupeeScan in your research or project, please star the repository! ⭐
