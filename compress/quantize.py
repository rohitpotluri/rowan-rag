import os
import torch
from transformers import AutoModelForCausalLM

# === Absolute Paths relative to this script ===
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "..", "models")
PRUNED_DIR      = os.path.join(MODELS_DIR, "pruned_mistral_7b")
QUANT_DIR       = os.path.join(MODELS_DIR, "final_quantized")

os.makedirs(QUANT_DIR, exist_ok=True)

def get_dir_size_mb(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

# 1) Size before quantization
size_before = get_dir_size_mb(PRUNED_DIR)
print(f"Model size before quantization: {size_before:.2f} MB")

# 2) Load pruned model
print("Loading pruned model for quantization...")
model = AutoModelForCausalLM.from_pretrained(PRUNED_DIR, torch_dtype=torch.float32)

# 3) Move to GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for quantization...")
    model = torch.nn.DataParallel(model)
model = model.to(device)

# 4) Quantize to 16-bit
print("Quantizing model to 16-bit (FP16)...")
model.half()

# 5) Save quantized model
print(f"Saving quantized model to '{QUANT_DIR}'...")
if isinstance(model, torch.nn.DataParallel):
    model.module.save_pretrained(QUANT_DIR)
else:
    model.save_pretrained(QUANT_DIR)

# 6) Unload and free memory
print("Unloading model and clearing GPU cache...")
del model
torch.cuda.empty_cache()

# 7) Size after quantization
size_after = get_dir_size_mb(QUANT_DIR)
print(f"Model size after quantization: {size_after:.2f} MB")
print(f"Size reduced by: {size_before - size_after:.2f} MB")
