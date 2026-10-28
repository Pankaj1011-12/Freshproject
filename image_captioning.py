"""
Image Captioning Module using BLIP (Bootstrapping Language-Image Pre-training)
Model: Salesforce/blip-image-captioning-base
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

def get_captioning_model(model_name="Salesforce/blip-image-captioning-base", device=None):
    """
    Lazy loader for BLIP Captioning model and processor to optimize memory usage.
    """
    global _PROCESSOR, _MODEL, _DEVICE
    if device is not None:
        _DEVICE = device

    if _PROCESSOR is None or _MODEL is None:
        print(f"[*] Loading BLIP Captioning model '{model_name}' on {_DEVICE}...")
        start_time = time.time()
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _PROCESSOR = BlipProcessor.from_pretrained(model_name)
        _MODEL = BlipForConditionalGeneration.from_pretrained(model_name).to(_DEVICE)
        _MODEL.eval()
        print(f"[+] Model loaded in {time.time() - start_time:.2f}s")

    return _PROCESSOR, _MODEL, _DEVICE

def clean_directive_prompt(prompt):
    """
    Checks if a prompt is an instruction (e.g. 'tell me about this image', 'describe image')
    rather than a sentence starter (e.g. 'a photo of').
    """
    if not prompt or not isinstance(prompt, str):
        return None, False
    p = prompt.strip()
    p_lower = p.lower().rstrip(":.?!")
    
    directive_prefixes = [
        "tell me about this image", "tell me about the image", "tell me about this picture",
        "tell me about image", "tell me about",
        "describe this image", "describe the image", "describe this picture",
        "describe image", "describe",
        "what is this image", "what is this picture", "what is in this image",
        "what is in the image", "what do you see in this image", "what do you see",
        "explain this image", "explain the image", "explain image", "explain",
        "what is this", "summarize image", "analyze this image", "analyze image"
    ]
    for d in directive_prefixes:
        if p_lower == d or p_lower.startswith(d + ":") or p_lower.startswith(d + " "):
            return None, True
    return p, False

def format_caption_sentence(caption, was_directive=False):
    """
    Formats the raw model caption into a clean, well-punctuated natural sentence.
    """
    if not caption:
        return ""
    caption = caption.strip()
    if was_directive:
        # Prepend natural English phrasing if it was an instruction
        if not caption.lower().startswith(("this image", "a photo", "an image", "the photo")):
            caption = f"This image shows {caption}"
    # Capitalize first letter
    if caption:
        caption = caption[0].upper() + caption[1:]
    # Ensure ending period
    if caption and not caption.endswith((".", "!", "?")):
        caption += "."
    return caption

def generate_caption(image_input, prompt=None, max_new_tokens=40, device=None):
    """
    Generate an image caption.
    Args:
        image_input: File path (str) or PIL Image object.
        prompt: Optional text prompt for conditional captioning.
        max_new_tokens: Maximum tokens to generate.
        device: 'cuda' or 'cpu' (default: auto).
    Returns:
        dict with 'caption', 'conditional', and 'elapsed_sec'.
    """
    processor, model, dev = get_captioning_model(device=device)

    # Clean conversational directive prompts so they don't awkwardly repeat in the output
    filtered_prompt, is_directive = clean_directive_prompt(prompt)

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
        if filtered_prompt:
            # Conditional image captioning with sentence prefix
            inputs = processor(raw_image, text=filtered_prompt, return_tensors="pt").to(dev)
        else:
            # Unconditional image captioning
            inputs = processor(images=raw_image, return_tensors="pt").to(dev)

        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3, repetition_penalty=1.2)
        raw_caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()

    final_caption = format_caption_sentence(raw_caption, was_directive=is_directive)
    elapsed = time.time() - start_time
    return {
        "caption": final_caption,
        "conditional": bool(filtered_prompt),
        "prompt": prompt,
        "elapsed_sec": round(elapsed, 3)
    }

def main():
    parser = argparse.ArgumentParser(description="Generate captions for images using BLIP")
    parser.add_argument("--image", type=str, default="your_image.jpg", help="Path to input image")
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt for conditional captioning")
    parser.add_argument("--max_tokens", type=int, default=35, help="Max tokens to generate")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        # Check parent or subfolder if nested
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
    print(f"       BLIP IMAGE CAPTIONING")
    print(f"========================================")
    print(f"Image   : {image_path}")
    print(f"Prompt  : {args.prompt if args.prompt else '(Unconditional - None)'}")
    print(f"Device  : {_DEVICE}")
    print(f"Running inference...\n")

    result = generate_caption(image_path, prompt=args.prompt, max_new_tokens=args.max_tokens)
    print(f"----------------------------------------")
    print(f"Result  : {result['caption']}")
    print(f"Latency : {result['elapsed_sec']} seconds")
    print(f"========================================\n")

if __name__ == "__main__":
    main()
