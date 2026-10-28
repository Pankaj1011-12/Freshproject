"""
Mini Multimodal Transformer - Main Project Entry Point
Provides a unified CLI, interactive menu, and automated test suite.
"""

import os
import sys
import argparse
import subprocess
import time

# Reconfigure stdout/stderr for Windows UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch

def banner():
    print(r"""
====================================================================
    MINI MULTIMODAL TRANSFORMER: VISION + LANGUAGE SYSTEM
    Image Captioning * Visual Question Answering * CLIP + T5
====================================================================
    """)

def check_environment():
    print("[*] Performing System & Environment Diagnostics...")
    print(f"    - Python Version      : {sys.version.split()[0]}")
    print(f"    - PyTorch Version     : {torch.__version__}")
    print(f"    - CUDA Available      : {torch.cuda.is_available()}")
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"    - Compute Device      : {device}")
    
    import transformers
    import PIL
    print(f"    - Transformers Version: {transformers.__version__}")
    print(f"    - Pillow Version      : {PIL.__version__}")

    sample_img = "your_image.jpg"
    if os.path.exists(sample_img):
        print(f"    - Sample Image        : Found ('{sample_img}')")
    else:
        print(f"    - Sample Image        : Not found, will be auto-generated")
    print("[+] Environment check passed!\n")

def run_tests():
    banner()
    print("[*] Running Full Project Test Suite...\n")
    check_environment()

    # Test 1: Image Captioning
    print("--------------------------------------------------------------------")
    print(">> [TEST 1/3] Testing Image Captioning (Salesforce/blip-image-captioning-base)...")
    res1 = subprocess.run([sys.executable, "image_captioning.py", "--image", "your_image.jpg"], capture_output=True, text=True, encoding="utf-8")
    if res1.returncode != 0:
        print(f"[!] Test 1 failed: {res1.stderr}")
        return False
    # Extract result line
    for line in res1.stdout.splitlines():
        if "Result  :" in line:
            print(f"   [PASS] {line.strip()}")
        elif "Latency :" in line:
            print(f"          {line.strip()}")

    # Test 2: Visual Question Answering
    print("--------------------------------------------------------------------")
    print(">> [TEST 2/3] Testing VQA (Salesforce/blip-vqa-base)...")
    q = "How many birds are there?"
    res2 = subprocess.run([sys.executable, "vqa.py", "--image", "your_image.jpg", "--question", q], capture_output=True, text=True, encoding="utf-8")
    if res2.returncode != 0:
        print(f"[!] Test 2 failed: {res2.stderr}")
        return False
    for line in res2.stdout.splitlines():
        if "Answer  :" in line:
            print(f"   [PASS] Q: '{q}' -> {line.strip()}")
        elif "Latency :" in line:
            print(f"          {line.strip()}")

    # Test 3: Advanced CLIP + T5
    print("--------------------------------------------------------------------")
    print(">> [TEST 3/3] Testing Advanced CLIP Feature Extraction & MLP Bridge...")
    res3 = subprocess.run([sys.executable, "advanced_clip_t5.py", "--image", "your_image.jpg"], capture_output=True, text=True, encoding="utf-8")
    if res3.returncode != 0:
        print(f"[!] Test 3 failed: {res3.stderr}")
        return False
    for line in res3.stdout.splitlines():
        if "Feature Vector Shape:" in line or "Feature Vector Norm :" in line:
            print(f"   [PASS] {line.strip()}")
        elif "Multimodal Output :" in line:
            print(f"   [PASS] {line.strip()}")

    print("--------------------------------------------------------------------")
    print("[+] ALL TESTS PASSED SUCCESSFULLY! The multimodal transformer is ready to use.\n")
    return True

def interactive_menu():
    banner()
    check_environment()

    while True:
        print("Please choose an option to run:")
        print("  [1] Image Captioning (BLIP)")
        print("  [2] Visual Question Answering (BLIP VQA)")
        print("  [3] Advanced CLIP + T5 Multimodal Transformer")
        print("  [4] Launch Interactive Web Application (Browser UI)")
        print("  [5] Run All System Self-Tests")
        print("  [0] Exit")
        
        try:
            choice = input("\nEnter choice [0-5]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == "1":
            img = input("Enter image path [default: your_image.jpg]: ").strip() or "your_image.jpg"
            prompt = input("Enter optional prompt (e.g. 'a photo of') [default: none]: ").strip() or None
            from image_captioning import generate_caption
            try:
                res = generate_caption(img, prompt=prompt)
                print(f"\n>> Caption: {res['caption']} (latency: {res['elapsed_sec']}s)\n")
            except Exception as e:
                print(f"[!] Error: {e}\n")

        elif choice == "2":
            img = input("Enter image path [default: your_image.jpg]: ").strip() or "your_image.jpg"
            q = input("Enter your question [default: 'What is in the image?']: ").strip() or "What is in the image?"
            from vqa import answer_question
            try:
                res = answer_question(img, q)
                print(f"\n>> Answer: {res['answer']} (latency: {res['elapsed_sec']}s)\n")
            except Exception as e:
                print(f"[!] Error: {e}\n")

        elif choice == "3":
            print("\nRunning Advanced CLIP + T5 module...")
            from advanced_clip_t5 import main as clip_main
            sys.argv = ["advanced_clip_t5.py", "--demo_train"]
            try:
                clip_main()
            except Exception as e:
                print(f"[!] Error: {e}\n")

        elif choice == "4":
            print("\nLaunching Web Application on http://127.0.0.1:5000 ...")
            from app import app
            app.run(host="0.0.0.0", port=5000, debug=False)
            break

        elif choice == "5":
            run_tests()

        elif choice in ("0", "exit", "quit", "q"):
            print("Goodbye!")
            break
        else:
            print("[!] Invalid option, please enter a number from 0 to 5.\n")

def main():
    parser = argparse.ArgumentParser(description="Mini Multimodal Transformer Suite")
    parser.add_argument("--mode", type=str, choices=["caption", "vqa", "clip-t5", "web", "test", "menu"], default="menu",
                        help="Execution mode: caption, vqa, clip-t5, web, test, or menu")
    parser.add_argument("--image", type=str, default="your_image.jpg", help="Path to input image")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for captioning or CLIP-T5")
    parser.add_argument("--question", type=str, default="What is in the image?", help="Question for VQA")
    parser.add_argument("--port", type=int, default=5000, help="Port for web app")
    parser.add_argument("--demo_train", action="store_true", help="Run demo training loop for CLIP-T5")
    args = parser.parse_args()

    if args.mode == "menu":
        if sys.stdin.isatty():
            interactive_menu()
        else:
            run_tests()
    elif args.mode == "test":
        run_tests()
    elif args.mode == "caption":
        from image_captioning import generate_caption
        res = generate_caption(args.image, prompt=args.prompt)
        print(f"\nGenerated Caption: {res['caption']} (in {res['elapsed_sec']}s)")
    elif args.mode == "vqa":
        from vqa import answer_question
        res = answer_question(args.image, args.question)
        print(f"\nVQA Question: {args.question}")
        print(f"VQA Answer  : {res['answer']} (in {res['elapsed_sec']}s)")
    elif args.mode == "clip-t5":
        from advanced_clip_t5 import main as clip_main
        sys.argv = ["advanced_clip_t5.py", "--image", args.image]
        if args.prompt:
            sys.argv.extend(["--prompt", args.prompt])
        if args.demo_train:
            sys.argv.append("--demo_train")
        clip_main()
    elif args.mode == "web":
        from app import app
        print(f"Starting web server at http://127.0.0.1:{args.port} ...")
        app.run(host="0.0.0.0", port=args.port, debug=False)

if __name__ == "__main__":
    main()
