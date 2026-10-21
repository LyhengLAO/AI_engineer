"""
evaluate.py
-----------
Évalue un modèle GGUF sur le set de validation Text-to-SQL.

Métriques:
  - exact_match : requête identique à la référence après normalisation
                  (casse, espaces, ';' final). Stricte : deux requêtes
                  équivalentes mais écrites différemment comptent comme fausses.
  - sql_valide  : la requête s'exécute en EXPLAIN sur une base SQLite en
                  mémoire créée à partir du schéma (colonnes/tables existantes,
                  syntaxe correcte). Ne prouve pas que le résultat est juste.

⚠️ RESSOURCE: 100% CPU, fonctionne en local sur AMD via llama-cpp-python.
   Pour un vrai avant/après, convertir aussi le modèle de base en GGUF
   (sans adapters) et relancer le script dessus.

Usage:
    python src/evaluate.py --model models/model-q4_k_m.gguf \
        --val_file data/processed/val.jsonl --out docs/results_finetuned.json
"""

import argparse
import json
import sqlite3
from pathlib import Path

from llama_cpp import Llama

from prompts import build_prompt, normalize_sql


def load_val_set(path, limit=None):
    with open(path, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    return examples[:limit] if limit else examples


def is_valid_sql(query: str, context: str) -> bool:
    """Vrai si la requête passe EXPLAIN sur une base SQLite construite depuis le schéma."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(context)
        conn.execute(f"EXPLAIN {query.strip().rstrip(';')}")
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def evaluate(llm, examples):
    n_exact, n_valid = 0, 0
    errors = []

    for i, ex in enumerate(examples):
        output = llm(
            build_prompt(ex["question"], ex["context"]),
            max_tokens=200,
            stop=["<|im_end|>"],
            temperature=0.0,
        )
        pred = output["choices"][0]["text"].strip()

        exact = normalize_sql(pred) == normalize_sql(ex["output"])
        valid = is_valid_sql(pred, ex["context"])
        n_exact += exact
        n_valid += valid
        if not exact and len(errors) < 10:
            errors.append({"question": ex["question"], "attendu": ex["output"], "predit": pred})

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(examples)} exemples évalués...")

    n = len(examples)
    return {
        "n_exemples": n,
        "exact_match_pct": round(n_exact / n * 100, 1),
        "sql_valide_pct": round(n_valid / n * 100, 1),
        "exemples_erreurs": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limiter le nombre d'exemples (évaluation rapide)")
    parser.add_argument("--out", type=str, default=None,
                        help="Chemin JSON pour sauvegarder les résultats")
    args = parser.parse_args()

    print(f"Chargement de {args.model}...")
    llm = Llama(model_path=args.model, n_ctx=1024,
                n_threads=args.n_threads, verbose=False)

    examples = load_val_set(args.val_file, args.limit)
    print(f"Évaluation sur {len(examples)} exemples...")
    results = evaluate(llm, examples)

    print(f"\n=== Résultats ({results['n_exemples']} exemples) ===")
    print(f"Exact match : {results['exact_match_pct']}%")
    print(f"SQL valide  : {results['sql_valide_pct']}%")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Résultats sauvegardés dans {args.out}")


if __name__ == "__main__":
    main()
