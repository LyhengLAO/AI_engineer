from langchain_core.tools import tool
from .base import register


@register
@tool
def compte_mots(texte: str) -> str:
    """Compte le nombre de mots dans un texte."""
    return str(len(texte.split()))


@register
@tool
def inverse_texte(texte: str) -> str:
    """Inverse une chaîne de caractères."""
    return texte[::-1]
