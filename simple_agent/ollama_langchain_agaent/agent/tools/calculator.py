from langchain_core.tools import tool
from .base import register


@register
@tool
def calculatrice(expression: str) -> str:
    """Évalue une expression mathématique simple, ex: '12*7+3'."""
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "Erreur : expression invalide (seuls les chiffres et + - * / ( ) sont autorisés)"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Erreur de calcul : {exc}"
