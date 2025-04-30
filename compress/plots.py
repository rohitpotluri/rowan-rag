#plots scaled histogram for pruned model.
import os
import gc
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM

# === Paths relative to this script ===
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "..", "models")
SAVE_DIR_BEFORE = os.path.join(MODELS_DIR, "mistral_original")
SAVE_DIR_AFTER  = os.path.join(MODELS_DIR, "pruned_mistral_7b")
PLOTS_DIR       = os.path.join(BASE_DIR, "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# 1) compute 99th‐percentile x‐limit from the uncompressed model
print("Computing x‐axis limit from uncompressed model...")
model_before = AutoModelForCausalLM.from_pretrained(SAVE_DIR_BEFORE, torch_dtype=torch.float32)
weights_before = []
for _, param in model_before.named_parameters():
    if "weight" in _ and param.requires_grad:
        weights_before.append(param.data.detach().cpu().view(-1))
all_np_before = torch.cat(weights_before).numpy()
limit = np.percentile(np.abs(all_np_before), 99)

# clean up
del model_before, weights_before, all_np_before
gc.collect()
torch.cuda.empty_cache()

# 2) plot scaled histogram for the pruned model
print("Plotting scaled distribution for pruned model...")
model_after = AutoModelForCausalLM.from_pretrained(SAVE_DIR_AFTER, torch_dtype=torch.float32)
weights_after = []
for _, param in model_after.named_parameters():
    if "weight" in _ and param.requires_grad:
        weights_after.append(param.data.detach().cpu().view(-1))
all_np_after = torch.cat(weights_after).numpy()

# percentage‐of‐total
pct_after = np.ones_like(all_np_after) / len(all_np_after) * 100

plt.hist(
    all_np_after,
    bins=100,
    weights=pct_after,
    color="grey",
    range=(-limit, limit)
)
plt.ylim(0, 2.5)
plt.hlines(2.5, -limit, limit, linestyles="dotted")
plt.title("Weight Distribution After Pruning (Scaled)")
plt.xlabel("Weight Value")
plt.ylabel("Percentage of weights (%)")
plt.savefig(os.path.join(PLOTS_DIR, "pruned_scaled_plot.png"))
plt.show()

# clean up
del model_after, weights_after, all_np_after, pct_after
gc.collect()
torch.cuda.empty_cache()
