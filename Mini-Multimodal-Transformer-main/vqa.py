"""
Visual Question Answering (VQA) Module using BLIP
Model: Salesforce/blip-vqa-base
"""

import os
import sys
import time
import argparse

# UTF-8 stdout configuration for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from PIL import Image

_PROCESSOR = None
_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def unload_model():
    """
    Explicitly unload the model to release memory.
    """
    global _PROCESSOR, _MODEL
    _PROCESSOR = None
    _MODEL = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_vqa_model(model_name="Salesforce/blip-vqa-base", device=None):
    """
    Lazy loader for BLIP VQA model and processor.
    """
    global _PROCESSOR, _MODEL, _DEVICE
    if device is not None:
        _DEVICE = device

    if _PROCESSOR is None or _MODEL is None:
        print(f"[*] Loading BLIP VQA model '{model_name}' on {_DEVICE}...")
        start_time = time.time()
        from transformers import BlipProcessor, BlipForQuestionAnswering
        _PROCESSOR = BlipProcessor.from_pretrained(model_name)
        _MODEL = BlipForQuestionAnswering.from_pretrained(model_name).to(_DEVICE)
        _MODEL.eval()
        print(f"[+] Model loaded in {time.time() - start_time:.2f}s")

    return _PROCESSOR, _MODEL, _DEVICE

def answer_question(image_input, question, max_new_tokens=35, device=None):
    """
    Answer a natural language question about an image.
    Args:
        image_input: File path (str) or PIL Image object.
        question: Question string (e.g. 'How many birds are there?').
        max_new_tokens: Maximum tokens to generate.
        device: 'cuda' or 'cpu' (default: auto).
    Returns:
        dict with 'answer', 'question', and 'elapsed_sec'.
    """
    if not question or not str(question).strip():
        raise ValueError("A valid question string must be provided.")

    processor, model, dev = get_vqa_model(device=device)

    # Load image if file path is provided
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        raw_image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        raw_image = image_input.convert("RGB")
    else:
        raise ValueError("image_input must be a file path string or PIL Image object")

    start_time = time.time()
    with torch.no_grad():
        inputs = processor(images=raw_image, text=question.strip(), return_tensors="pt").to(dev)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        answer = processor.decode(output_ids[0], skip_special_tokens=True).strip()

    elapsed = time.time() - start_time
    return {
        "answer": answer,
        "question": question.strip(),
        "elapsed_sec": round(elapsed, 3)
    }

def main():
    parser = argparse.ArgumentParser(description="Visual Question Answering using BLIP")
    parser.add_argument("--image", type=str, default="your_image.jpg", help="Path to input image")
    parser.add_argument("--question", type=str, default="What is in the image?", help="Question to ask about image")
    parser.add_argument("--interactive", action="store_true", help="Start interactive question session")
    parser.add_argument("--max_tokens", type=int, default=35, help="Max tokens to generate")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        alt_path = os.path.join(os.path.dirname(__file__), image_path)
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            print(f"[!] Warning: Image '{image_path}' not found.")
            print("[*] Creating a default test image 'your_image.jpg'...")
            from PIL import ImageDraw
            img = Image.new("RGB", (300, 200), color=(70, 130, 180))
            d = ImageDraw.Draw(img)
            d.rectangle([50, 50, 250, 150], fill=(240, 230, 140))
            img.save("your_image.jpg")
            image_path = "your_image.jpg"

    print(f"\n========================================")
    print(f"       BLIP VISUAL QUESTION ANSWERING")
    print(f"========================================")
    print(f"Image   : {image_path}")
    print(f"Device  : {_DEVICE}")

    # Preload model
    get_vqa_model()

    if args.interactive:
        print("\n[!] Interactive mode started. Type 'quit' or 'exit' to stop.")
        while True:
            try:
                user_q = input("\nEnter your question: ").strip()
                if not user_q or user_q.lower() in ("quit", "exit", "q"):
                    print("Exiting interactive mode.")
                    break
                res = answer_question(image_path, user_q, max_new_tokens=args.max_tokens)
                print(f"Answer  : {res['answer']}  ({res['elapsed_sec']}s)")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break
    else:
        print(f"Question: {args.question}")
        print(f"Running inference...\n")
        res = answer_question(image_path, args.question, max_new_tokens=args.max_tokens)
        print(f"----------------------------------------")
        print(f"Answer  : {res['answer']}")
        print(f"Latency : {res['elapsed_sec']} seconds")
        print(f"========================================\n")

if __name__ == "__main__":
    main()
