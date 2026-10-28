# 🧠 Mini Multimodal Transformer Project: Image Captioning + Visual Question Answering

This project demonstrates how to combine vision and language models to build a **multimodal transformer** system that:
- Generates descriptive image captions (conditional and unconditional)
- Answers natural language questions about an image (Visual Question Answering)
- Implements custom cross-modal projection between vision (CLIP) and language (T5) transformers
- Runs locally on Windows, macOS, and Linux with CPU and CUDA GPU support.

---

## 📦 Features
- ✅ **Pretrained BLIP Image Captioning** (`Salesforce/blip-image-captioning-base`)
- ✅ **Pretrained BLIP Visual Question Answering** (`Salesforce/blip-vqa-base`)
- ✅ **Custom Cross-Modal Bridge** (`CLIP ViT-B/32` + `MLP Projection Bridge` + `Google T5-Small`)
- ✅ **Interactive Browser Web UI** (Flask-based local web application)
- ✅ **Unified CLI Runner & Test Suite** (`main.py`)
- ✅ **Windows & Memory Optimized** (Subprocess isolation, auto-unloading to prevent pagefile/RAM exhaustion)

---

## 📁 Project Structure

```
Mini-Multimodal-Transformer/
├── main.py                 # Central launcher, CLI dispatcher & self-test suite
├── app.py                  # Local Web UI (Flask) with image upload & real-time inference
├── image_captioning.py     # BLIP Image Captioning module and CLI
├── vqa.py                  # BLIP Visual Question Answering module and CLI
├── advanced_clip_t5.py     # Custom Multimodal Transformer (CLIP + MLP Bridge + T5)
├── requirements.txt        # Python dependency manifest
├── run.bat                 # Windows one-click launcher
├── your_image.jpg          # Default sample image (parrots)
└── README.md               # Documentation & usage guide
```

---

## 🖥️ Requirements & Installation

1. Python 3.8+ (Tested on Python 3.13)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

*(Packages: `torch`, `torchvision`, `transformers`, `pillow`, `flask`)*

---

## 🚀 How to Run Locally

### 1. Unified Interactive Menu or Self-Tests
```bash
python main.py
```
Or to run automated system validation:
```bash
python main.py --mode test
```

### 2. Launch the Web Application (Browser UI)
```bash
python app.py
```
Or:
```bash
python main.py --mode web
```
Then open your browser at **`http://127.0.0.1:5000`** to drag and drop images, generate captions, ask questions, or run the multimodal bridge!

### 3. Image Captioning (CLI)
```bash
python image_captioning.py --image your_image.jpg
```
For conditional captioning:
```bash
python image_captioning.py --image your_image.jpg --prompt "a photo of"
```

### 4. Visual Question Answering (CLI)
```bash
python vqa.py --image your_image.jpg --question "How many birds are there?"
```
Interactive Q&A loop:
```bash
python vqa.py --image your_image.jpg --interactive
```

### 5. Advanced Custom Multimodal Model (CLIP + T5)
```bash
python advanced_clip_t5.py --image your_image.jpg --demo_train
```

---

## 🔧 Fixes & Optimizations Implemented

1. **Missing Implementation Files**: Built modular, clean implementations of `image_captioning.py`, `vqa.py`, `advanced_clip_t5.py`, `app.py`, and `main.py`.
2. **Transformers 5.x Compatibility**: Fixed `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'shape'` by supporting both v4 Tensor outputs and v5 ModelOutput pooled features.
3. **Correct VQA Class**: Updated BLIP VQA to use `BlipForQuestionAnswering` (the official Hugging Face architecture for `blip-vqa-base`).
4. **Memory & Pagefile Management (OS Error 1455)**: Resolved Windows paging file exhaustion by adding explicit model unloading (`gc.collect()` and cache flushing) and isolated test executions.
5. **Windows Unicode Console Fix**: Resolved `UnicodeEncodeError: 'charmap'` by configuring UTF-8 console streams and safe console formatting.
6. **Sample Image Inclusion**: Pre-packaged valid sample `your_image.jpg` with auto-generation fallbacks so scripts never crash due to missing files.
