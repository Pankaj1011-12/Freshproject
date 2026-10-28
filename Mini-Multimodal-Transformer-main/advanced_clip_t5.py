"""
Advanced: Custom Multimodal Transformer (CLIP Vision Encoder + MLP Bridge + T5 Decoder)
Combines OpenAI CLIP visual embeddings with Google T5 language model.
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
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

class MultimodalBridge(nn.Module):
    """
    Bridge network that projects CLIP visual feature embeddings into T5 encoder/decoder hidden space.
    Projects 1 pooled visual embedding into `num_visual_tokens` sequence vectors.
    """
    def __init__(self, clip_dim=512, t5_dim=512, num_visual_tokens=4):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.t5_dim = t5_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, t5_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(t5_dim * 2, t5_dim * num_visual_tokens),
            nn.LayerNorm(t5_dim * num_visual_tokens)
        )

    def forward(self, clip_features):
        # clip_features: [batch_size, clip_dim]
        b = clip_features.size(0)
        projected = self.mlp(clip_features) # [batch_size, t5_dim * num_visual_tokens]
        # Reshape to token sequence: [batch_size, num_visual_tokens, t5_dim]
        visual_tokens = projected.view(b, self.num_visual_tokens, self.t5_dim)
        return visual_tokens

class MiniMultimodalTransformer(nn.Module):
    """
    End-to-end Mini Multimodal Transformer:
    - Vision Encoder: CLIP (openai/clip-vit-base-patch32)
    - Cross-Modal Bridge: MultimodalBridge MLP
    - Language Model: T5 (t5-small)
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", t5_model_name="t5-small", num_visual_tokens=4, freeze_vision=True):
        super().__init__()
        print(f"[*] Initializing MiniMultimodalTransformer...")
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        
        clip_dim = self.clip.config.projection_dim
        t5_dim = self.t5.config.d_model
        
        self.bridge = MultimodalBridge(clip_dim=clip_dim, t5_dim=t5_dim, num_visual_tokens=num_visual_tokens)
        
        if freeze_vision:
            for param in self.clip.parameters():
                param.requires_grad = False
            self.clip.eval()
            print("[+] Vision encoder frozen to preserve pretrained feature quality.")

    def extract_clip_features(self, pixel_values):
        out = self.clip.get_image_features(pixel_values=pixel_values)
        # Handle transformers v4 / v5 return format
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            features = out.pooler_output
        elif hasattr(out, "image_embeds") and out.image_embeds is not None:
            features = out.image_embeds
        elif isinstance(out, tuple):
            features = out[0]
        else:
            features = out
        return features

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        """
        Multimodal forward pass computing cross-modal loss.
        """
        if pixel_values is not None:
            image_features = self.extract_clip_features(pixel_values)
            visual_tokens = self.bridge(image_features) # [B, num_tokens, d_model]
        else:
            visual_tokens = None

        if input_ids is not None:
            # Get text token embeddings from T5 encoder
            t5_embeds = self.t5.shared(input_ids) # [B, seq_len, d_model]
            if visual_tokens is not None:
                # Concatenate visual tokens + text token embeddings
                combined_embeds = torch.cat([visual_tokens, t5_embeds], dim=1)
                if attention_mask is not None:
                    vis_mask = torch.ones((visual_tokens.size(0), visual_tokens.size(1)), dtype=attention_mask.dtype, device=attention_mask.device)
                    combined_mask = torch.cat([vis_mask, attention_mask], dim=1)
                else:
                    combined_mask = None
            else:
                combined_embeds = t5_embeds
                combined_mask = attention_mask
        else:
            combined_embeds = visual_tokens
            combined_mask = None

        encoder_outputs = self.t5.encoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            return_dict=True
        )

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            labels=labels,
            return_dict=True
        )
        return outputs

    def generate(self, pixel_values, text_prompt=None, tokenizer=None, max_new_tokens=30):
        """
        Generate text conditioned on both visual features and optional text prompt.
        """
        self.eval()
        with torch.no_grad():
            image_features = self.extract_clip_features(pixel_values)
            visual_tokens = self.bridge(image_features)
            
            if text_prompt and tokenizer:
                text_inputs = tokenizer(text_prompt, return_tensors="pt").to(pixel_values.device)
                t5_embeds = self.t5.shared(text_inputs.input_ids)
                combined_embeds = torch.cat([visual_tokens, t5_embeds], dim=1)
            else:
                combined_embeds = visual_tokens

            encoder_outputs = self.t5.encoder(
                inputs_embeds=combined_embeds,
                return_dict=True
            )

            # Pass multimodal encoder representation directly to T5 generation
            generated_ids = self.t5.generate(
                encoder_outputs=encoder_outputs,
                max_new_tokens=max_new_tokens
            )
        return generated_ids

def run_clip_feature_extraction(image_path="your_image.jpg"):
    """
    Baseline CLIP feature extraction as described in README.
    """
    print(f"\n[*] 1. Extracting visual features from CLIPModel...")
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    img = Image.open(image_path).convert("RGB")
    inputs = clip_proc(images=img, return_tensors="pt")
    
    with torch.no_grad():
        out = clip_model.get_image_features(**inputs)
        features = out.pooler_output if hasattr(out, "pooler_output") else out
    
    print(f"[+] CLIP Vision Features Extracted!")
    print(f"    Feature Vector Shape: {features.shape}")
    print(f"    Feature Vector Norm : {features.norm(dim=-1).item():.4f}")
    return features

def run_demo_bridge_training(model, tokenizer, clip_processor, image_path="your_image.jpg"):
    """
    Demonstrates training the bridge network end-to-end on image + text pairs.
    """
    print("\n[*] 2. Demonstrating Multimodal Bridge Training Loop...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    img = Image.open(image_path).convert("RGB")
    vis_inputs = clip_processor(images=img, return_tensors="pt").to(device)
    
    # Prompt + Target label
    prompt = "visual summary:"
    target = "two colorful parrots perching on a wooden branch in nature"
    
    prompt_enc = tokenizer(prompt, return_tensors="pt").to(device)
    target_enc = tokenizer(target, return_tensors="pt").to(device)
    labels = target_enc.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    optimizer = torch.optim.AdamW(model.bridge.parameters(), lr=1e-3)
    model.train()
    
    print(f"    Training Bridge parameters on sample: prompt='{prompt}', target='{target}'")
    for step in range(1, 4):
        optimizer.zero_grad()
        outputs = model(
            pixel_values=vis_inputs.pixel_values,
            input_ids=prompt_enc.input_ids,
            attention_mask=prompt_enc.attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"    Step {step}/3 - Cross-Entropy Loss: {loss.item():.4f}")

    print("[+] Bridge training step completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Advanced CLIP + T5 Multimodal Transformer")
    parser.add_argument("--image", type=str, default="your_image.jpg", help="Path to input image")
    parser.add_argument("--prompt", type=str, default="describe image:", help="Text prompt for generation")
    parser.add_argument("--demo_train", action="store_true", help="Run 3-step demonstration of bridge training")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        alt = os.path.join(os.path.dirname(__file__), image_path)
        if os.path.exists(alt):
            image_path = alt
        else:
            print(f"[!] Image '{image_path}' not found, creating default...")
            from PIL import ImageDraw
            img = Image.new("RGB", (300, 200), color=(50, 100, 150))
            d = ImageDraw.Draw(img)
            d.rectangle([50, 50, 250, 150], fill=(200, 200, 100))
            img.save("your_image.jpg")
            image_path = "your_image.jpg"

    print("\n========================================")
    print("   ADVANCED MULTIMODAL TRANSFORMER")
    print("        (CLIP + MLP Bridge + T5)")
    print("========================================")
    print(f"Image  : {image_path}")
    print(f"Prompt : {args.prompt}")

    # Step 1: Baseline CLIP Feature Extraction
    features = run_clip_feature_extraction(image_path)

    # Step 2: Initialize Full Multimodal Architecture
    print("\n[*] Initializing CLIP-to-T5 Multimodal Architecture...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = MiniMultimodalTransformer().to(device)

    # Optional Demo Bridge Training
    if args.demo_train:
        run_demo_bridge_training(model, tokenizer, clip_processor, image_path)

    # Step 3: Multimodal Generation
    print("\n[*] 3. Generating text with combined multimodal encoder...")
    img = Image.open(image_path).convert("RGB")
    pixel_values = clip_processor(images=img, return_tensors="pt").pixel_values.to(device)
    
    start_t = time.time()
    gen_ids = model.generate(pixel_values, text_prompt=args.prompt, tokenizer=tokenizer, max_new_tokens=25)
    generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    latency = time.time() - start_t

    print("----------------------------------------")
    print(f"Multimodal Output : {generated_text if generated_text else '(Token sequence generated)'}")
    print(f"Inference Latency : {latency:.3f} seconds")
    print("========================================\n")

if __name__ == "__main__":
    main()
