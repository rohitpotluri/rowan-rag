#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, "..", "models")
TOKENIZER_SRC = "mistralai/Mistral-7B-v0.1"
OUT_FILE      = os.path.join(BASE_DIR, "compare_output.txt")

# Model variants: (label, path, dtype)
MODELS = [
    ("Final Quantized", os.path.join(MODELS_DIR, "final_quantized"),   torch.float16),
    ("Pruned-only",     os.path.join(MODELS_DIR, "final_quantized"),   torch.float16),
    ("Original",        os.path.join(MODELS_DIR, "mistral_original"),  torch.float32),
]

# Prompt and generation settings
PROMPT = "How is new year celebrated in the United States?"
GEN_KWARGS = {
    "max_new_tokens":      50,
    "num_beams":           4,
    "no_repeat_ngram_size": 2,
    "early_stopping":      True,
    # pad_token_id set after tokenizer load
}

def main():
    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SRC)
    tokenizer.pad_token = tokenizer.eos_token
    GEN_KWARGS["pad_token_id"] = tokenizer.eos_token_id

    with open(OUT_FILE, "w", encoding="utf-8") as out:
        out.write(f"Prompt: {PROMPT!r}\n\n")
        for label, path, dtype in MODELS:
            out.write(f"{label}:\n")

            # Load model onto GPUs
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
                device_map="auto"
            )
            model.eval()

            # Tokenize and move inputs
            inputs = tokenizer(PROMPT, return_tensors="pt", padding=True).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(**inputs, **GEN_KWARGS)

            # Decode just the answer (includes prompt by default)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            out.write(text + "\n\n")

            # Cleanup GPU for next model
            del model, inputs, outputs
            torch.cuda.empty_cache()

    print(f"All outputs saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
