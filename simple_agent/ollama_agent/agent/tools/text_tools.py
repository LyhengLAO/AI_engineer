from .base import register_tool


@register_tool(
    name="compte_mots",
    description="Compte le nombre de mots dans un texte. Ex : compte_mots(bonjour le monde)",
)
def word_count(text: str) -> str:
    return str(len(text.split()))


@register_tool(
    name="inverse_texte",
    description="Inverse une chaîne de caractères. Ex : inverse_texte(bonjour)",
)
def reverse_text(text: str) -> str:
    return text[::-1]
