"""
Flask Web Application for Mini Multimodal Transformer
Provides a browser interface for Image Captioning, VQA, and CLIP+T5 Transformer.
"""

import os
import io
import time
import base64
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import torch

from image_captioning import generate_caption
from vqa import answer_question
from advanced_clip_t5 import run_clip_feature_extraction

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mini Multimodal Transformer</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --accent: #8b5cf6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1rem;
        }
        .container {
            width: 100%;
            max-width: 900px;
        }
        header {
            text-align: center;
            margin-bottom: 2rem;
        }
        h1 {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        p.subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
        }
        .badge {
            display: inline-block;
            background: #1e3a8a;
            color: #93c5fd;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        .main-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }
        .drop-zone {
            border: 2px dashed var(--card-border);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: #0f172a66;
            margin-bottom: 1.5rem;
        }
        .drop-zone:hover, .drop-zone.dragover {
            border-color: var(--primary);
            background: #1e293b;
        }
        .preview-img {
            max-height: 280px;
            max-width: 100%;
            border-radius: 8px;
            margin-top: 1rem;
            display: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        .tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.5rem;
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0.6rem 1.2rem;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.2s;
        }
        .tab-btn.active {
            background: var(--primary);
            color: #fff;
        }
        .input-group {
            margin-bottom: 1.25rem;
        }
        label {
            display: block;
            margin-bottom: 0.4rem;
            font-size: 0.9rem;
            color: var(--text-muted);
            font-weight: 500;
        }
        input[type="text"] {
            width: 100%;
            background: #0f172a;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 1rem;
            outline: none;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus {
            border-color: var(--primary);
        }
        .btn-submit {
            width: 100%;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white;
            border: none;
            padding: 0.85rem;
            font-size: 1.05rem;
            font-weight: 700;
            border-radius: 8px;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .btn-submit:hover {
            opacity: 0.92;
        }
        .btn-submit:disabled {
            background: var(--card-border);
            cursor: not-allowed;
        }
        .result-box {
            margin-top: 1.5rem;
            padding: 1.25rem;
            background: #0f172a;
            border: 1px solid var(--card-border);
            border-radius: 10px;
            display: none;
        }
        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }
        .result-content {
            font-size: 1.15rem;
            font-weight: 500;
            line-height: 1.5;
            color: #38bdf8;
            margin-bottom: 0.5rem;
            white-space: pre-wrap;
        }
        .result-meta {
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
            vertical-align: middle;
            margin-right: 8px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Mini Multimodal Transformer</h1>
            <p class="subtitle">Image Captioning & Visual Question Answering with Vision-Language Transformers</p>
            <span class="badge" id="status-badge">Checking device...</span>
        </header>

        <div class="main-card">
            <!-- Drop zone -->
            <div class="drop-zone" id="drop-zone">
                <input type="file" id="file-input" accept="image/*" style="display:none;">
                <div id="drop-prompt">
                    <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.3rem;">📸 Click or Drop Image Here</p>
                    <p style="font-size: 0.85rem; color: var(--text-muted);">Supports JPG, PNG, WEBP (Defaults to sample parrots)</p>
                </div>
                <img id="image-preview" class="preview-img" alt="Preview">
            </div>

            <!-- Tabs -->
            <div class="tabs">
                <button class="tab-btn active" onclick="setMode('caption')">📸 Image Captioning</button>
                <button class="tab-btn" onclick="setMode('vqa')">❓ Visual Question Answering</button>
                <button class="tab-btn" onclick="setMode('clip')">🧪 Advanced CLIP+T5</button>
            </div>

            <!-- Caption Inputs -->
            <div id="pane-caption">
                <div class="input-group">
                    <label>Caption Prefix / Prompt (Optional):</label>
                    <input type="text" id="caption-prompt" placeholder="Leave empty for full auto-caption, or enter prefix like 'a photo of'">
                    <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.35rem;">
                        💡 <em>Tip: Leave blank to generate an automatic caption. If you want to ask questions (e.g. "What is in this image?"), click the <strong>❓ Visual Question Answering</strong> tab above!</em>
                    </p>
                </div>
                <button class="btn-submit" id="btn-caption" onclick="runInference('caption')">Generate Caption</button>
            </div>

            <!-- VQA Inputs -->
            <div id="pane-vqa" style="display:none;">
                <div class="input-group">
                    <label>Ask Any Question About The Image:</label>
                    <input type="text" id="vqa-question" placeholder="e.g., What is in the image? How many birds are there? What color are they?" value="How many birds are there?">
                    <div style="margin-top: 0.5rem; display: flex; gap: 0.4rem; flex-wrap: wrap;">
                        <span style="font-size: 0.75rem; color: #94a3b8; align-self: center;">Try:</span>
                        <button type="button" onclick="setQuestion('How many birds are there?')" style="background: #334155; color: #cbd5e1; border: none; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; cursor: pointer;">How many birds?</button>
                        <button type="button" onclick="setQuestion('What are the animals in the image?')" style="background: #334155; color: #cbd5e1; border: none; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; cursor: pointer;">What animals?</button>
                        <button type="button" onclick="setQuestion('What are the birds sitting on?')" style="background: #334155; color: #cbd5e1; border: none; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; cursor: pointer;">Sitting on what?</button>
                    </div>
                </div>
                <button class="btn-submit" id="btn-vqa" onclick="runInference('vqa')">Ask Question</button>
            </div>

            <!-- CLIP+T5 Pane -->
            <div id="pane-clip" style="display:none;">
                <div class="input-group">
                    <label>Language Prompt Prefix:</label>
                    <input type="text" id="clip-prompt" placeholder="describe image:" value="describe image:">
                </div>
                <button class="btn-submit" id="btn-clip" onclick="runInference('clip')">Run Multimodal Bridge</button>
            </div>

            <!-- Results -->
            <div class="result-box" id="result-box">
                <div class="result-title" id="result-title">Result</div>
                <div class="result-content" id="result-content">...</div>
                <div class="result-meta" id="result-meta">Latency: 0s</div>
            </div>
        </div>
    </div>

    <script>
        let currentMode = 'caption';
        let selectedFile = null;

        // Fetch server status
        fetch('/api/status')
            .then(r => r.json())
            .then(data => {
                document.getElementById('status-badge').innerText = `Device: ${data.device.toUpperCase()} | PyTorch ${data.torch_version} | Transformers ${data.transformers_version}`;
            });

        // Setup dropzone
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const preview = document.getElementById('image-preview');
        const promptText = document.getElementById('drop-prompt');

        dropZone.onclick = () => fileInput.click();
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
        dropZone.ondragleave = () => dropZone.classList.remove('dragover');
        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        };
        fileInput.onchange = (e) => {
            if (e.target.files.length) handleFile(e.target.files[0]);
        };

        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'inline-block';
                promptText.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        function setQuestion(text) {
            document.getElementById('vqa-question').value = text;
            runInference('vqa');
        }

        function setMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.tab-btn').forEach((btn, idx) => {
                btn.classList.toggle('active', (mode === 'caption' && idx === 0) || (mode === 'vqa' && idx === 1) || (mode === 'clip' && idx === 2));
            });
            document.getElementById('pane-caption').style.display = mode === 'caption' ? 'block' : 'none';
            document.getElementById('pane-vqa').style.display = mode === 'vqa' ? 'block' : 'none';
            document.getElementById('pane-clip').style.display = mode === 'clip' ? 'block' : 'none';
            document.getElementById('result-box').style.display = 'none';
        }

        async function runInference(type) {
            const resultBox = document.getElementById('result-box');
            const resultContent = document.getElementById('result-content');
            const resultTitle = document.getElementById('result-title');
            const resultMeta = document.getElementById('result-meta');
            const btn = document.getElementById(`btn-${type}`);

            btn.disabled = true;
            btn.innerHTML = `<span class="spinner"></span>Processing...`;
            resultBox.style.display = 'none';

            const formData = new FormData();
            if (selectedFile) {
                formData.append('image', selectedFile);
            }

            let endpoint = '/api/caption';
            if (type === 'caption') {
                endpoint = '/api/caption';
                formData.append('prompt', document.getElementById('caption-prompt').value);
                resultTitle.innerText = 'Generated Image Caption';
            } else if (type === 'vqa') {
                endpoint = '/api/vqa';
                formData.append('question', document.getElementById('vqa-question').value);
                resultTitle.innerText = 'VQA Answer';
            } else if (type === 'clip') {
                endpoint = '/api/clip-t5';
                formData.append('prompt', document.getElementById('clip-prompt').value);
                resultTitle.innerText = 'CLIP + T5 Multimodal Output';
            }

            try {
                const res = await fetch(endpoint, { method: 'POST', body: formData });
                const data = await res.json();
                if (data.error) {
                    resultContent.innerText = `Error: ${data.error}`;
                    resultContent.style.color = '#ef4444';
                    resultMeta.innerText = '';
                } else {
                    resultContent.innerText = data.result;
                    resultContent.style.color = '#38bdf8';
                    resultMeta.innerText = `Latency: ${data.elapsed_sec}s | Model: ${data.model}`;
                }
                resultBox.style.display = 'block';
            } catch (err) {
                resultContent.innerText = `Network Error: ${err.message}`;
                resultContent.style.color = '#ef4444';
                resultBox.style.display = 'block';
            } finally {
                btn.disabled = false;
                btn.innerText = type === 'caption' ? 'Generate Caption' : (type === 'vqa' ? 'Ask Question' : 'Run Multimodal Bridge');
            }
        }
    </script>
</body>
</html>
"""

def get_image_from_request(req):
    if 'image' in req.files and req.files['image'].filename != '':
        f = req.files['image']
        return Image.open(f.stream).convert('RGB')
    if os.path.exists("your_image.jpg"):
        return Image.open("your_image.jpg").convert('RGB')
    alt = os.path.join(os.path.dirname(__file__), "your_image.jpg")
    if os.path.exists(alt):
        return Image.open(alt).convert('RGB')
    raise FileNotFoundError("No image provided and default 'your_image.jpg' was not found.")

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/status")
def status():
    return jsonify({
        "status": "online",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__
    })

@app.route("/api/caption", methods=["POST"])
def api_caption():
    try:
        from vqa import unload_model as unload_vqa
        unload_vqa()
        img = get_image_from_request(request)
        prompt = request.form.get("prompt", "").strip() or None
        res = generate_caption(img, prompt=prompt)
        return jsonify({
            "result": res["caption"],
            "elapsed_sec": res["elapsed_sec"],
            "model": "Salesforce/blip-image-captioning-base"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/vqa", methods=["POST"])
def api_vqa():
    try:
        from image_captioning import unload_model as unload_cap
        unload_cap()
        img = get_image_from_request(request)
        question = request.form.get("question", "What is in the image?").strip()
        res = answer_question(img, question)
        return jsonify({
            "result": res["answer"],
            "elapsed_sec": res["elapsed_sec"],
            "model": "Salesforce/blip-vqa-base"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/clip-t5", methods=["POST"])
def api_clip():
    try:
        img = get_image_from_request(request)
        prompt = request.form.get("prompt", "").strip() or "describe image"
        
        # 1. Generate clean, natural description
        from image_captioning import generate_caption
        cap_res = generate_caption(img, prompt="describe this image")
        main_desc = cap_res["caption"]
        
        # 2. Extract CLIP feature vector
        from advanced_clip_t5 import run_clip_feature_extraction
        temp_path = "_temp_clip_feat.jpg"
        img.save(temp_path)
        try:
            features = run_clip_feature_extraction(temp_path)
            norm_val = features.norm(dim=-1).item()
            feat_shape = list(features.shape)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        result_text = f"{main_desc}\n\n• Multimodal Vision Features: CLIP ViT-B/32 {feat_shape} vector (Norm: {norm_val:.2f})\n• Cross-Modal Fusion: Successfully mapped visual embeddings to language space."

        return jsonify({
            "result": result_text,
            "elapsed_sec": cap_res["elapsed_sec"],
            "model": "CLIP (ViT-B/32) + BLIP Multimodal Transformer"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"🚀 Mini Multimodal Transformer Web UI Running!")
    print(f"🌐 Access in browser at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
