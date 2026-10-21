"""
download_dataset.py
-------------------
Télécharge le dataset réel `b-mc2/sql-create-context` (HuggingFace Hub,
~78k exemples question + schéma CREATE TABLE -> requête SQL), en prend un
sous-échantillon reproductible et le convertit au format du projet.

Ressource: 🖥️ Local CPU (ou Colab). Téléchargement ~ quelques dizaines de Mo,
aucun GPU nécessaire. Nécessite une connexion internet.

Usage:
    python src/download_dataset.py --n_train 2000 --n_val 200 --out data/processed
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from prompts import build_instruction

DATASET_ID = "b-mc2/sql-create-context"


def to_record(row: dict) -> dict:
    """Champs d'origine conservés (question/context) pour l'évaluation."""
    return {
        "instruction": build_instruction(row["question"], row["context"]),
        "output": row["answer"].strip(),
        "question": row["question"].strip(),
        "context": row["context"].strip(),
    }


def write_jsonl(records, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/processed")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Téléchargement de {DATASET_ID} depuis HuggingFace...")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"Dataset complet: {len(ds)} exemples, colonnes: {ds.column_names}")

    needed = args.n_train + args.n_val
    if needed > len(ds):
        raise ValueError(f"Demandé {needed} exemples, dataset de {len(ds)}")

    ds = ds.shuffle(seed=args.seed).select(range(needed))
    val_rows = ds.select(range(args.n_val))
    train_rows = ds.select(range(args.n_val, needed))

    write_jsonl((to_record(r) for r in train_rows), out_dir / "train.jsonl")
    write_jsonl((to_record(r) for r in val_rows), out_dir / "val.jsonl")

    print(f"Train: {args.n_train} exemples -> {out_dir / 'train.jsonl'}")
    print(f"Val:   {args.n_val} exemples -> {out_dir / 'val.jsonl'}")


if __name__ == "__main__":
    main()
