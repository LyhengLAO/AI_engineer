"""
inference.py
------------
Inférence locale du modèle fine-tuné quantifié (GGUF), 100% CPU.
Fonctionne sur AMD sans aucun souci (pas de dépendance CUDA/ROCm).

Prérequis:
    pip install llama-cpp-python

Usage:
    python src/inference.py --model models/model-q4_k_m.gguf \
        --context "CREATE TABLE employes (nom VARCHAR, salaire INTEGER, poste VARCHAR)" \
        --question "Quel est le salaire moyen des Data Scientists ?"
"""

import argparse

from llama_cpp import Llama

from prompts import build_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/model-q4_k_m.gguf")
    parser.add_argument("--context", type=str, required=True,
                        help="Schéma: une ou plusieurs instructions CREATE TABLE")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--n_ctx", type=int, default=1024)
    parser.add_argument("--n_threads", type=int, default=8,
                        help="Nombre de threads CPU (ajuste selon ton processeur AMD)")
    args = parser.parse_args()

    print(f"Chargement du modèle {args.model} (CPU, {args.n_threads} threads)...")
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        verbose=False,
    )

    output = llm(
        build_prompt(args.question, args.context),
        max_tokens=200,
        stop=["<|im_end|>"],
        temperature=0.0,
    )

    print("\n--- Requête SQL générée ---")
    print(output["choices"][0]["text"].strip())


if __name__ == "__main__":
    main()
