from langchain_core.tools import tool
from .base import register


@register
@tool
def recherche_web(query: str) -> str:
    """Recherche une information récente sur le web (actualités, prix, faits récents)."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Erreur : installe d'abord le package avec 'pip install ddgs'"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "Aucun résultat trouvé."
        lines = [f"- {r.get('title', '')} : {r.get('body', '')[:200]}" for r in results]
        return "\n".join(lines)
    except Exception as exc:
        return f"Erreur de recherche : {exc}"
