from .base import register_tool


@register_tool(
    name="calculatrice",
    description="Évalue une expression mathématique simple. Ex : calculatrice(12*7+3)",
)
def calculator(expression: str) -> str:
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return "Erreur : expression invalide (seuls les chiffres et + - * / ( ) sont autorisés)"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Erreur de calcul : {exc}"
