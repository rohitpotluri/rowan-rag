import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM

# === Absolute Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
PLOTS_DIR = os.path.join(BASE_DIR, "..", "plots")
SAVE_DIR_BEFORE = os.path.join(MODELS_DIR, "mistral_original")
SAVE_DIR_AFTER = os.path.join(MODELS_DIR, "pruned_mistral_7b")

os.makedirs(SAVE_DIR_BEFORE, exist_ok=True)
os.makedirs(SAVE_DIR_AFTER, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Step 1: Load model (download from HF)
print("Downloading and loading Mistral 7B model...")
model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.save_pretrained(SAVE_DIR_BEFORE)

# Move to multi-GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for pruning...")
    model = torch.nn.DataParallel(model)
model = model.to(device)

# Step 2: Prune %
prune_percent = float(input("\nEnter % of smallest magnitude weights to prune per layer (recommended 10): "))

# Step 3: Layerwise Approximate Pruning
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

# Step 4: Save pruned model
print("\nSaving pruned model...")
model.module.save_pretrained(SAVE_DIR_AFTER) if isinstance(model, torch.nn.DataParallel) else model.save_pretrained(SAVE_DIR_AFTER)

# Step 5: Plot weight distributions (same sampled weights and same y-scale)
sample_size = 500_000
torch.manual_seed(42)

print("Loading original model for plotting...")
model_before = AutoModelForCausalLM.from_pretrained(SAVE_DIR_BEFORE, torch_dtype=torch.float32).to("cuda")
print("Sampling weights before pruning...")

sampled_before = []
shared_indices = {}
for name, param in model_before.named_parameters():
    if 'weight' in name and param.requires_grad:
        weight_flat = param.data.view(-1)
        n = min(sample_size, weight_flat.numel())
        idx = torch.randperm(weight_flat.numel(), device=weight_flat.device)[:n]
        sampled = weight_flat[idx]
        sampled_before.append(sampled)
        shared_indices[name] = idx
sampled_before_np = torch.cat(sampled_before).detach().cpu().numpy()

print("Loading pruned model for plotting...")
model_after = AutoModelForCausalLM.from_pretrained(SAVE_DIR_AFTER, torch_dtype=torch.float32).to("cuda")
print("Sampling weights after pruning...")

sampled_after = []
for name, param in model_after.named_parameters():
    if 'weight' in name and param.requires_grad:
        weight_flat = param.data.view(-1)
        idx = shared_indices.get(name)
        if idx is not None and idx.numel() <= weight_flat.numel():
            sampled = weight_flat[idx]
            sampled_after.append(sampled)
sampled_after_np = torch.cat(sampled_after).detach().cpu().numpy()

print("Generating zoomed-in histograms...")
max_y = max(
    np.histogram(sampled_before_np, bins=200, range=(-0.02, 0.02))[0].max(),
    np.histogram(sampled_after_np, bins=200, range=(-0.02, 0.02))[0].max()
)

plt.clf()
plt.hist(sampled_before_np, bins=200, range=(-0.02, 0.02), color='blue')
plt.ylim(0, max_y + 0.1 * max_y)
plt.title('Weight Distribution Before Pruning')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.savefig(os.path.join(PLOTS_DIR, 'histogram_before_pruning.png'))
plt.show()

plt.clf()
plt.hist(sampled_after_np, bins=200, range=(-0.02, 0.02), color='green')
plt.ylim(0, max_y + 0.1 * max_y)
plt.title('Weight Distribution After Pruning')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.savefig(os.path.join(PLOTS_DIR, 'histogram_after_pruning.png'))
plt.show()

# Step 6: Free up GPU
del model_before
torch.cuda.empty_cache()
print("\n[Cleanup complete: freed GPU memory used by original model]")
