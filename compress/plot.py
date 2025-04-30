#plots histograms for compressed and uncompressed models
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

def plot_weight_distribution(model_path, title, output_file, color):
    # load and move to CPU
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model = model.to("cpu")

    # collect all weights
    all_weights = []
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            all_weights.append(param.data.view(-1))
    all_weights_np = torch.cat(all_weights).numpy()

    # compute percentage per weight
    pct = np.ones_like(all_weights_np) / len(all_weights_np) * 100
    # dynamic x-range (99th percentile)
    limit = np.percentile(np.abs(all_weights_np), 99)

    # plot
    plt.hist(
        all_weights_np,
        bins=100,
        weights=pct,
        color=color,
        range=(-limit, limit)
    )
    plt.title(title)
    plt.xlabel("Weight Value")
    plt.ylabel("Percentage of weights (%)")
    plt.savefig(os.path.join(PLOTS_DIR, output_file))
    plt.show()

    # clean up
    del model, all_weights, all_weights_np, pct
    gc.collect()
    torch.cuda.empty_cache()

# plot before-pruning
print("Plotting distribution BEFORE pruning...")
plot_weight_distribution(
    SAVE_DIR_BEFORE,
    "Weight Distribution Before Pruning",
    "histogram_before_pruning.png",
    "gray"
)

# plot after-pruning
print("Plotting distribution AFTER pruning...")
plot_weight_distribution(
    SAVE_DIR_AFTER,
    "Weight Distribution After Pruning",
    "histogram_after_pruning.png",
    "black"
)
