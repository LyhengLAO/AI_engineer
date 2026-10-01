# langchain-ollama-agent

Version **LangChain / LangGraph** de l'agent tool-calling, basée sur [Ollama](https://ollama.com) (100% local et gratuit). C'est la variante "outils du marché" du repo [ollama-agent](../ollama-agent) fait from scratch : même idée, mais avec les briques standard utilisées en entreprise.

## Différence avec la version from-scratch

| | From scratch | Cette version |
|---|---|---|
| Boucle agent | Codée à la main (regex ACTION/FINAL) | `create_react_agent` (LangGraph) |
| Appel d'outils | Parsing texte | Tool calling structuré natif (JSON) |
| Outils | Fonctions Python simples | `@tool` LangChain (`BaseTool`) |
| Écosystème | Aucun | Compatible RAG, mémoire, LangSmith, etc. |

Les deux gardent la même philosophie : **une architecture à plugins** pour ajouter des outils sans toucher au reste du code.

## Installation

```bash
git clone <url-du-repo>
cd langchain-ollama-agent

python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

Installer Ollama et un modèle qui supporte le **tool calling** (llama3.1, qwen2.5, mistral-nemo...) :

```bash
ollama pull llama3.1
ollama serve
```

## Utilisation

```bash
python main.py                                   # mode interactif
python main.py -q "Combien fait 15*8 ?"           # question directe
python main.py --model qwen2.5                    # autre modèle
```

## Architecture

```
langchain-ollama-agent/
├── agent/
│   ├── builder.py          # Construction du graphe LangGraph (ChatOllama + tools)
│   └── tools/
│       ├── base.py         # Registre TOOLS + décorateur @register
│       ├── __init__.py     # Auto-découverte des outils
│       ├── calculator.py
│       ├── text_tools.py
│       └── web_search.py   # Recherche web via le package ddgs
├── tests/
├── main.py                 # CLI
└── requirements.txt
```

## Scaler : ajouter un outil

```python
# agent/tools/meteo.py
from langchain_core.tools import tool
from .base import register

@register
@tool
def meteo(ville: str) -> str:
    """Donne la météo actuelle d'une ville."""
    return f"Il fait beau à {ville}"
```

Rien d'autre à modifier : `agent/tools/__init__.py` importe automatiquement
tous les fichiers du dossier, l'outil est immédiatement disponible pour
l'agent.

## Pistes d'évolution (pertinentes marché GenAI)

- **RAG** : brancher un `Chroma` retriever comme outil (`create_retriever_tool`) — cohérent avec un projet RAG existant (Ollama + ChromaDB + BM25 + reranker)
- **Mémoire de conversation** : `checkpointer` LangGraph (ex: `MemorySaver`) pour garder l'historique entre appels
- **Observabilité** : brancher LangSmith (gratuit en dev) pour tracer les appels d'outils et debug les chaînes de raisonnement
- **API** : exposer `build_agent()` derrière FastAPI, dockerisé, pour une démo déployable en entretien
- **Évaluation** : RAGAS pour scorer la pertinence/fidélité des réponses si un outil RAG est ajouté

## Tests

```bash
pytest
```
