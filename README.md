# 🧪 LabPPE AI

**Automated Personal Protective Equipment compliance verification for biotechnology laboratories, powered by computer vision.**

LabPPE AI is a web application that verifies whether a person is wearing the required PPE before entering a biotech laboratory. It uses a fine-tuned YOLO detector to identify PPE in an image or webcam capture, generates a compliance report, and issues an access decision with a spoken announcement.

![LabPPE AI dashboard](assets/screenshot.png)

---

## Features

- **Four-item PPE detection**: lab coat, gloves, safety goggles, face mask
- **Two input modes**: image upload or live webcam capture
- **Bounding-box visualization** with per-detection confidence scores
- **Colour-coded compliance checklist** (green = present, red = missing)
- **Binary access decision**: 🟢 GRANTED / 🔴 DENIED with compliance percentage
- **Spoken announcement** of the result via the browser's Web Speech API
- **Downloadable compliance report** in Markdown and CSV
- **CPU-optimized** — runs on any modern laptop, no GPU required

---

## Tech stack

| Layer | Choice | Reason |
|---|---|---|
| Detection model | YOLO11n fine-tuned on CPPE-5 | Nano architecture, best CPU accuracy/speed trade-off |
| Inference runtime | OpenVINO FP16 | ~3–4× CPU speedup over PyTorch on Intel silicon |
| UI | Streamlit | Rapid prototyping, native webcam support |
| Vision | Ultralytics, OpenCV, Pillow | Standard, well-maintained stack |
| Voice | Web Speech API | Client-side, offline, zero dependencies |

---

## Quick start

**Prerequisites**: Python 3.11+

```bash
git clone https://github.com/<your-username>/labppe-ai.git
cd labppe-ai
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Train the detector** (one-time, ~30–60 min on CPU):

```bash
python train.py --batch 16 --workers 4 --cache ram
```

Training auto-downloads the CPPE-5 dataset from Hugging Face, fine-tunes with a frozen backbone, and exports an OpenVINO model for fast CPU inference.

**Launch the app**:

```bash
streamlit run app.py
```

Then open <http://localhost:8501>.

---

## How it works

1. A person captures or uploads an image at the lab entrance.
2. The YOLO detector locates PPE items in the frame.
3. A rule-based policy maps raw detections to required PPE (e.g. a coverall satisfies "lab coat"; goggles *or* a face shield satisfies "eye protection").
4. Compliance score = (detected required PPE / total required PPE) × 100.
5. Access is granted only if the score reaches 100%.
6. The system announces the decision aloud and generates a downloadable report.

---

## Project structure

```
labppe-ai/
├── app.py               Streamlit dashboard
├── train.py             CPU-adapted training + OpenVINO export
├── evaluate.py          mAP + CPU latency benchmark
├── src/
│   ├── config.py        PPE policy, model paths, thresholds
│   ├── detector.py      Format-agnostic YOLO wrapper
│   ├── compliance.py    Checklist and access decision (pure logic)
│   ├── report.py        Compliance report generation
│   └── voice.py         Spoken announcement composition
├── requirements.txt
├── models/              Trained weights and exported artifacts (gitignored)
├── data/                CPPE-5 dataset, downloaded on first run (gitignored)
└── reports/             Saved compliance reports (gitignored)
```

---

## Model and dataset

- **Base model**: YOLO11n (~2.6 M parameters), pretrained on COCO
- **Dataset**: [CPPE-5](https://huggingface.co/datasets/rishitdagli/cppe-5) — 5 PPE classes (Coverall, Face_Shield, Gloves, Goggles, Mask)
- **Training strategy**: transfer learning with frozen backbone (`freeze=10`) at 416×416, early stopping on validation loss
- **Post-training**: FP16 OpenVINO export for CPU-accelerated inference

Reproduce the benchmark on your own hardware with:

```bash
python evaluate.py
```

This reports mAP50 / mAP50-95 on the validation split and side-by-side inference latency for PyTorch and OpenVINO formats.

---

## Design notes

- **Decoupled logic**: `detector.py` is the only module that imports Ultralytics; the UI and compliance policy operate on plain `Detection` objects, keeping the model swappable and the logic trivially testable.
- **Policy as data**: PPE requirements live in `src/config.py`, not in code. Adding a new required item is a one-line change.
- **Graceful fallback**: the detector chooses OpenVINO → trained PyTorch → pretrained COCO in priority order, so the app launches usefully even before training completes.
- **Format-agnostic reporting**: reports are generated as Markdown *and* CSV for both human review and downstream analytics.

---

## Acknowledgments

Developed by **Team Synergy** during the **Astra Program**, a joint academic initiative between Romanian and American universities focused on interdisciplinary innovation in artificial intelligence and applied sciences.

Dataset citation:

> Dagli, R., & Shaikh, A. M. (2021). *CPPE-5: Medical Personal Protective Equipment Dataset.* [arXiv:2112.09569](https://arxiv.org/abs/2112.09569)

---

## License

Released under the MIT License. See `LICENSE` for details.