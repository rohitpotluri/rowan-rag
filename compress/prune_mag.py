import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
model_id = "mistralai/Mistral-7B-v0.1"
save_dir_before = os.path.join("..", "models", "mistral_original")
save_dir_after = os.path.join("..", "models", "pruned_mistral_7b")
plot_dir = os.path.join("..", "plots")
os.makedirs(save_dir_before, exist_ok=True)
os.makedirs(save_dir_after, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Step 1: Load model (DOWNLOAD from Huggingface)
print("Downloading and loading Mistral 7B model...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.save_pretrained(save_dir_before)  # Save full original model before pruning
model = model.to("cuda")

# Step 2: Weight histogram before pruning
print("Calculating weight stats before pruning...")
all_weights_tensor = []
for name, param in model.named_parameters():
    if 'weight' in name and param.requires_grad:
        all_weights_tensor.append(param.data.view(-1))
all_weights_concat = torch.cat(all_weights_tensor)

# For plotting
all_weights_np = all_weights_concat.detach().cpu().numpy()

plt.hist(all_weights_np, bins=100, color='blue')
plt.title('Weight Distribution Before Pruning')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.savefig(os.path.join(plot_dir, 'histogram_before_pruning.png'))
plt.show()

print("\n=== Weight Statistics BEFORE Pruning ===")
print(f"Mean: {np.mean(all_weights_np):.6f}")
print(f"Std Dev: {np.std(all_weights_np):.6f}")
print(f"Min: {np.min(all_weights_np):.6f}")
print(f"Max: {np.max(all_weights_np):.6f}")
zero_ratio_before = np.sum(all_weights_np == 0) / len(all_weights_np) * 100
print(f"Zero Weights: {zero_ratio_before:.2f}%")

# Step 3: Prune %
prune_percent = float(input("\nEnter % of smallest magnitude weights to prune per layer (recommended 10): "))

# Step 4: Layerwise Approximate Pruning
print("\nStarting layerwise pruning with sampling...")
for name, param in model.named_parameters():
    if 'weight' in name and param.requires_grad:
        weight_abs = param.data.abs()
        num_elements = weight_abs.numel()

        if num_elements > 10_000_000:
            # Sample only 10 million values randomly
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

# Step 5: Weight histogram after pruning (using sampled weights)
print("Calculating weight stats after pruning...")
all_weights_sample_after = []
for name, param in model.named_parameters():
    if 'weight' in name and param.requires_grad:
        weight_flat = param.data.view(-1)
        sample_size = min(500_000, weight_flat.numel())
        indices = torch.randperm(weight_flat.numel(), device=weight_flat.device)[:sample_size]
        sampled = weight_flat[indices]
        all_weights_sample_after.append(sampled)

all_weights_concat_sample_after = torch.cat(all_weights_sample_after)

all_weights_np_after = all_weights_concat_sample_after.detach().cpu().numpy()

plt.hist(all_weights_np_after, bins=100, color='green')
plt.title('Weight Distribution After Pruning (Sampled)')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.savefig(os.path.join(plot_dir, 'histogram_after_pruning.png'))
plt.show()

print("\n=== Weight Statistics AFTER Pruning (sampled) ===")
print(f"Mean: {np.mean(all_weights_np_after):.6f}")
print(f"Std Dev: {np.std(all_weights_np_after):.6f}")
print(f"Min: {np.min(all_weights_np_after):.6f}")
print(f"Max: {np.max(all_weights_np_after):.6f}")
zero_ratio_after = np.sum(all_weights_np_after == 0) / len(all_weights_np_after) * 100
print(f"Zero Weights: {zero_ratio_after:.2f}%")

# Step 6: Save pruned model
print("\nSaving pruned model...")
model.save_pretrained(save_dir_after)

# Step 7: Model size comparison
def get_dir_size_mb(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)

size_before = get_dir_size_mb(save_dir_before)
size_after = get_dir_size_mb(save_dir_after)

print(f"\nModel size before pruning: {size_before:.2f} MB")
print(f"Model size after pruning: {size_after:.2f} MB")

# Step 8: Plot size comparison
plt.figure(figsize=(6, 4))
plt.bar(["Before Pruning", "After Pruning"], [size_before, size_after], color=["blue", "green"])
plt.title("Model Size Before vs After Pruning")
plt.ylabel("Size (MB)")
plt.savefig(os.path.join(plot_dir, 'pruning_size_comparison.png'))
plt.show()
