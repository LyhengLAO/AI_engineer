"""
prompts.py
----------
Format de prompt partagé entre l'entraînement, l'inférence, l'évaluation et
la démo. Une seule source de vérité : si le format change entre train et
inférence, le modèle fine-tuné perd en performance.

Tâche : Text-to-SQL. Entrée = schéma (CREATE TABLE) + question en langage
naturel. Sortie = requête SQL.
"""

import re

INSTRUCTION_TEMPLATE = (
    "Écris la requête SQL qui répond à la question, en utilisant "
    "uniquement le schéma fourni. Réponds uniquement avec la requête SQL.\n\n"
    "Schéma:\n{context}\n\nQuestion: {question}"
)


def build_instruction(question: str, context: str) -> str:
    return INSTRUCTION_TEMPLATE.format(context=context.strip(), question=question.strip())


def build_prompt(question: str, context: str) -> str:
    """Prompt d'inférence (ChatML, format natif de Qwen2.5). S'arrête avant la réponse."""
    return (
        f"<|im_start|>user\n{build_instruction(question, context)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_training_text(instruction: str, output: str) -> str:
    """Texte complet d'entraînement : prompt + réponse attendue."""
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def normalize_sql(sql: str) -> str:
    """Normalisation légère pour comparer deux requêtes (casse, espaces, ';' final)."""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()
