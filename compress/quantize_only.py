#!/usr/bin/env python3
import os
import shutil
import torch
from transformers import AutoModelForCausalLM

# === Paths relative to this script ===
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR        = os.path.join(BASE_DIR, "..", "models")
ORIG_DIR          = os.path.join(MODELS_DIR, "mistral_original")
FINAL_QUANT_DIR   = os.path.join(MODELS_DIR, "final_quantized")

def get_dir_size_mb(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

def main():
    # Remove old final_quantized if it exists
    if os.path.isdir(FINAL_QUANT_DIR):
        print(f"Removing existing '{FINAL_QUANT_DIR}'...")
        shutil.rmtree(FINAL_QUANT_DIR)

    # 1) Size before quantization
    size_before = get_dir_size_mb(ORIG_DIR)
    print(f"Model size before quantization: {size_before:.2f} MB")

    # 2) Load original model in FP32
    print("Loading original Mistral-7B model for FP16 quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        ORIG_DIR,
        torch_dtype=torch.float32,
    )

    # 3) Move to GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for quantization...")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # 4) Quantize weights to FP16
    print("Converting model weights to 16-bit (FP16)...")
    model.half()

    # 5) Save to final_quantized
    print(f"Saving FP16 model to '{FINAL_QUANT_DIR}'...")
    # ensure parent exists
    os.makedirs(FINAL_QUANT_DIR, exist_ok=True)
    if isinstance(model, torch.nn.DataParallel):
        model.module.save_pretrained(FINAL_QUANT_DIR)
    else:
        model.save_pretrained(FINAL_QUANT_DIR)

    # 6) Unload & clear GPU
    print("Unloading model and clearing GPU cache...")
    del model
    torch.cuda.empty_cache()

    # 7) Size after quantization
    size_after = get_dir_size_mb(FINAL_QUANT_DIR)
    print(f"Model size after quantization: {size_after:.2f} MB")
    print(f"Total reduction: {size_before - size_after:.2f} MB")

if __name__ == "__main__":
    main()
