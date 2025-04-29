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

# Step 1: Load model and move to GPU
print("Downloading and loading Mistral 7B model...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model = model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save original model
model.save_pretrained(save_dir_before)
tokenizer.save_pretrained(save_dir_before)

# Step 2: Weight histogram before pruning
all_weights = []
for name, param in model.named_parameters():
    if 'weight' in name and param.requires_grad:
        all_weights.append(param.data.detach().cpu().numpy().flatten())
all_weights_flat = np.concatenate(all_weights)

plt.hist(all_weights_flat, bins=100, color='blue')
plt.title('Weight Distribution Before Pruning')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.savefig(os.path.join(plot_dir, 'histogram_before_pruning.png'))
plt.show()

print("\n=== Weight Statistics BEFORE Pruning ===")
print(f"Mean: {np.mean(all_weights_flat):.6f}")
print(f"Std Dev: {np.std(all_weights_flat):.6f}")
print(f"Min: {np.min(all_weights_flat):.6f}")
print(f"Max: {np.max(all_weights_flat):.6f}")
zero_ratio_before = np.sum(all_weights_flat == 0) / len(all_weights_flat) * 100
print(f"Zero Weights: {zero_ratio_before:.2f}%")

# Step 3: Prune %
prune_percent = float(input("\nEnter % of smallest magnitude weights to prune (recommended 10): "))

# Step 4: Prune
thresholds = []
for name, param in model.named_parameters():
    if 'weight' in name and param.requires_grad:
        tensor = param.data
        threshold = torch.quantile(tensor.abs(), prune_percent / 100)
        mask = tensor.abs() > threshold
        param.data = tensor * mask
        thresholds.append((name, threshold.item()))

print(f"\nPruning complete. Applied threshold on weights based on {prune_percent}% smallest magnitudes.")

# Step 5: Weight histogram after pruning
all_weights_after = []
for name, param in model.named_parameters():
    if 'weight' in name and param.requires_grad:
        all_weights_after.append(param.data.detach().cpu().numpy().flatten())
all_weights_flat_after = np.concatenate(all_weights_after)

plt.hist(all_weights_flat_after, bins=100, color='green')
plt.title('Weight Distribution After Pruning')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.savefig(os.path.join(plot_dir, 'histogram_after_pruning.png'))
plt.show()

print("\n=== Weight Statistics AFTER Pruning ===")
print(f"Mean: {np.mean(all_weights_flat_after):.6f}")
print(f"Std Dev: {np.std(all_weights_flat_after):.6f}")
print(f"Min: {np.min(all_weights_flat_after):.6f}")
print(f"Max: {np.max(all_weights_flat_after):.6f}")
zero_ratio_after = np.sum(all_weights_flat_after == 0) / len(all_weights_flat_after) * 100
print(f"Zero Weights: {zero_ratio_after:.2f}%")

# Step 6: Save pruned model
model.save_pretrained(save_dir_after)
tokenizer.save_pretrained(save_dir_after)

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
