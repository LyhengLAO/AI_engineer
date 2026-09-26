# ollama-agent

Agent IA "from scratch" (pattern **ReAct**) basé sur [Ollama](https://ollama.com), 100% local et gratuit. Aucun framework type LangChain — juste `requests` et une architecture à plugins pour les outils.

## Installation

```bash
git clone <url-du-repo>
cd ollama-agent

python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

Installer et lancer Ollama :

```bash
# https://ollama.com/download
ollama pull llama3.1     # ou mistral, qwen2.5, etc.
ollama serve              # généralement lancé automatiquement après installation
```

## Utilisation

Mode interactif :

```bash
python main.py
```

Mode direct (une question) :

```bash
python main.py -q "Combien fait 15*8, puis compte les mots dans ta réponse ?"
```

Changer de modèle :

```bash
python main.py --model mistral
```

## Architecture

```
ollama-agent/
├── agent/
│   ├── core.py            # Boucle ReAct + parsing ACTION/FINAL
│   ├── ollama_client.py   # Client HTTP minimal vers l'API Ollama
│   └── tools/
│       ├── base.py        # Décorateur @register_tool + registre global
│       ├── __init__.py    # Auto-découverte des outils
│       ├── calculator.py
│       ├── text_tools.py
│       └── web_search.py
├── tests/
├── main.py                # CLI
└── requirements.txt
```

Le modèle raisonne étape par étape : à chaque tour, il répond soit par
`ACTION: nom_outil(entrée)` pour utiliser un outil, soit par `FINAL: ...`
pour donner sa réponse définitive. La boucle exécute l'outil, renvoie le
résultat au modèle, et recommence jusqu'à la réponse finale ou la limite
d'itérations.

## Scaler : ajouter un nouvel outil

Le système est conçu pour grossir facilement. Il suffit de créer un
fichier dans `agent/tools/` et de décorer une fonction :

```python
# agent/tools/meteo.py
from .base import register_tool

@register_tool(
    name="meteo",
    description="Donne la météo d'une ville. Ex : meteo(Paris)",
)
def get_weather(ville: str) -> str:
    # ta logique ici (appel API, etc.)
    return f"Il fait beau à {ville}"
```

Aucune autre modification n'est nécessaire : `agent/tools/__init__.py`
importe automatiquement tous les fichiers du dossier au démarrage, donc
l'outil apparaît immédiatement dans le prompt système et devient
utilisable par le modèle.

Idées d'outils à ajouter pour aller plus loin :
- lecture/écriture de fichiers locaux
- requêtes SQL sur une base de données
- appel à une API météo, bourse, etc.
- exécution de code Python dans un sandbox

## Autres pistes de scalabilité

- **Mémoire persistante** : sauvegarder l'historique des conversations dans un fichier JSON ou une base SQLite entre les sessions.
- **Plusieurs agents spécialisés** : créer une classe `Agent` par domaine (recherche, code, data) et un routeur qui choisit le bon agent selon la question.
- **Streaming** : passer `stream=True` dans `OllamaClient` pour afficher la réponse token par token.
- **Interface web** : brancher `agent.core.Agent` derrière une petite API FastAPI ou un artefact HTML.

## Tests

```bash
pytest
```

## Licence

Libre d'utilisation et de modification pour un usage personnel ou d'apprentissage.
