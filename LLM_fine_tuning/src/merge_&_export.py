"""
merge_and_export.py
--------------------
Fusionne les adapters LoRA avec le modèle de base pour obtenir un modèle
complet, prêt à être converti en GGUF via llama.cpp.

⚠️ RESSOURCE:
- Fusion (merge): peut se faire sur CPU mais demande de la RAM
  (~4-6 Go pour un modèle 1.5B en fp16). Faisable en local si ta RAM
  le permet, sinon fais-le sur Colab (plus rapide, pas de risque de OOM).
- La conversion en GGUF (étape suivante, via llama.cpp) se fait ensuite
  en CPU, en local, sans problème.

Usage:
    python src/merge_and_export.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --lora_adapters models/lora_adapters \
        --output_dir models/merged_model
"""

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora_adapters", type=str, default="models/lora_adapters")
    parser.add_argument("--output_dir", type=str, default="models/merged_model")
    args = parser.parse_args()

    print(f"Chargement du modèle de base {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Chargement des adapters LoRA depuis {args.lora_adapters}...")
    model = PeftModel.from_pretrained(base_model, args.lora_adapters)

    print("Fusion des poids LoRA dans le modèle de base...")
    model = model.merge_and_unload()

    print(f"Sauvegarde du modèle fusionné dans {args.output_dir}...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print("\nÉtape suivante — conversion en GGUF (en local, CPU, via llama.cpp):")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print("  cd llama.cpp && pip install -r requirements.txt")
    print(f"  python convert_hf_to_gguf.py ../{args.output_dir} "
          f"--outfile ../models/model-f16.gguf")
    print("  ./llama-quantize ../models/model-f16.gguf "
          "../models/model-q4_k_m.gguf Q4_K_M")


if __name__ == "__main__":
    main()
