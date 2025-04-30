import os 
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Absolute Paths relative to this script ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
PLOTS_DIR = os.path.join(BASE_DIR, "..", "plots")
SAVE_DIR_BEFORE = os.path.join(MODELS_DIR, "mistral_original")
SAVE_DIR_AFTER = os.path.join(MODELS_DIR, "pruned_mistral_7b")

os.makedirs(SAVE_DIR_BEFORE, exist_ok=True)
os.makedirs(SAVE_DIR_AFTER, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Step 1: Load model (DOWNLOAD from Huggingface)
print("Downloading and loading Mistral 7B model...")
model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Save full model before pruning
model.save_pretrained(SAVE_DIR_BEFORE)

# Move model to GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for pruning...")
    model = torch.nn.DataParallel(model)
model = model.to(device)

# Step 2: Weight histogram before pruning
print("Calculating weight stats before pruning...")
all_weights_tensor = []
for name, param in model.named_parameters():
    if 'weight' in name and param.requires_grad:
        all_weights_tensor.append(param.data.view(-1))
all_weights_concat = torch.cat(all_weights_tensor)
all_weights_np = all_weights_concat.detach().cpu().numpy()

# plot percentage histogram before pruning
pct_before = np.ones_like(all_weights_np) / len(all_weights_np) * 100
plt.hist(all_weights_np,
         bins=100,
         weights=pct_before,
         color='blue')
# zoom in around zero
abs_max_before = np.percentile(np.abs(all_weights_np), 99)
plt.xlim(-abs_max_before, abs_max_before)
plt.title('Weight Distribution Before Pruning')
plt.xlabel('Weight Value')
plt.ylabel('Percentage of weights (%)')
plt.savefig(os.path.join(PLOTS_DIR, 'histogram_before_pruning.png'))
plt.show()

print("\n=== Weight Statistics BEFORE Pruning ===")
print(f"Mean: {np.mean(all_weights_np):.6f}")
print(f"Std Dev: {np.std(all_weights_np):.6f}")
print(f"Min: {np.min(all_weights_np):.6f}")
print(f"Max: {np.max(all_weights_np):.6f}")
zero_ratio_before = np.sum(all_weights_np == 0) / len(all_weights_np) * 100
print(f"Zero Weights: {zero_ratio_before:.2f}%")

# free up the big pre-prune tensor
del all_weights_concat, all_weights_tensor
torch.cuda.empty_cache()

# Step 3: Prune %
prune_percent = float(input("\nEnter % of smallest magnitude weights to prune per layer (recommended 10): "))

# Step 4: Layerwise Approximate Pruning
print("\nStarting layerwise pruning with sampling...")
for name, param in model.named_parameters():
    if 'weight' in name and param.requires_grad:
        weight_abs = param.data.abs()
        num_elements = weight_abs.numel()

        if num_elements > 10_000_000:
            indices = torch.randperm(num_elements, device=weight_abs.device)[:10_000_000]
            sample = weight_abs.view(-1)[indices]
            threshold = torch.quantile(sample, prune_percent / 100)
            print(f"Layer {name}: sampled threshold {threshold.item():.6f}")
        else:
            threshold = torch.quantile(weight_abs.view(-1), prune_percent / 100)
            print(f"Layer {name}: full tensor threshold {threshold.item():.6f}")

        mask = weight_abs > threshold
        param.data.mul_(mask)

print(f"\nPruning complete. Applied {prune_percent}% smallest weights per layer individually (with sampling if needed).")

# Step 5: Weight histogram after pruning (using all weights)
print("Calculating weight stats after pruning...")
all_weights_tensor_after = []
for name, param in model.named_parameters():
    if 'weight' in name and param.requires_grad:
        all_weights_tensor_after.append(param.data.view(-1))
all_weights_concat_after = torch.cat(all_weights_tensor_after)
all_weights_np_after = all_weights_concat_after.detach().cpu().numpy()

# plot percentage histogram after pruning
pct_after = np.ones_like(all_weights_np_after) / len(all_weights_np_after) * 100
plt.hist(all_weights_np_after,
         bins=100,
         weights=pct_after,
         color='green')
# zoom in around zero
abs_max_after = np.percentile(np.abs(all_weights_np_after), 99)
plt.xlim(-abs_max_after, abs_max_after)
plt.title('Weight Distribution After Pruning')
plt.xlabel('Weight Value')
plt.ylabel('Percentage of weights (%)')
plt.savefig(os.path.join(PLOTS_DIR, 'histogram_after_pruning.png'))
plt.show()

print("\n=== Weight Statistics AFTER Pruning ===")
print(f"Mean: {np.mean(all_weights_np_after):.6f}")
print(f"Std Dev: {np.std(all_weights_np_after):.6f}")
print(f"Min: {np.min(all_weights_np_after):.6f}")
print(f"Max: {np.max(all_weights_np_after):.6f}")
zero_ratio_after = np.sum(all_weights_np_after == 0) / len(all_weights_np_after) * 100
print(f"Zero Weights: {zero_ratio_after:.2f}%")

# Step 6: Save pruned model
print("\nSaving pruned model...")
model.module.save_pretrained(SAVE_DIR_AFTER) if isinstance(model, torch.nn.DataParallel) else model.save_pretrained(SAVE_DIR_AFTER)

# Step 7: Model size comparison
def get_dir_size_mb(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

size_before = get_dir_size_mb(SAVE_DIR_BEFORE)
size_after = get_dir_size_mb(SAVE_DIR_AFTER)

print(f"\nModel size before pruning: {size_before:.2f} MB")
print(f"Model size after pruning: {size_after:.2f} MB")

# Step 8: Plot size comparison
plt.figure(figsize=(6, 4))
plt.bar(["Before Pruning", "After Pruning"], [size_before, size_after], color=["blue", "green"])
plt.title("Model Size Before vs After Pruning")
plt.ylabel("Size (MB)")
plt.savefig(os.path.join(PLOTS_DIR, 'pruning_size_comparison.png'))
plt.show()
