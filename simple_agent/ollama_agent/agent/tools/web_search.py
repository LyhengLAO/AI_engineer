from .base import register_tool


@register_tool(
    name="recherche_web",
    description="Recherche une information récente sur le web. Ex : recherche_web(capitale du Japon)",
)
def web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Erreur : installe d'abord le package avec 'pip install duckduckgo-search'"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "Aucun résultat trouvé."
        lines = [f"- {r.get('title', '')} : {r.get('body', '')[:200]}" for r in results]
        return "\n".join(lines)
    except Exception as exc:
        return f"Erreur de recherche : {exc}"
