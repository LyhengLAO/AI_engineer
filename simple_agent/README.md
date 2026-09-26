# Agent IA avec Ollama — 2 versions

Ce dépôt (ou ces deux dépôts) contient **le même agent IA**, implémenté de deux façons différentes, à des fins de portfolio :

1. **`ollama-agent`** — implémentation *from scratch* (sans framework)
2. **`langchain-ollama-agent`** — implémentation avec **LangChain / LangGraph**

Les deux font exactement la même chose : un agent qui reçoit une question, décide d'utiliser ou non des outils (calculatrice, recherche web, etc.), et répond — le tout en local et gratuit avec [Ollama](https://ollama.com).

## Pourquoi deux versions ?

- La version **from scratch** montre que tu comprends le fonctionnement interne d'un agent (boucle de raisonnement, parsing des décisions du modèle, prompt engineering).
- La version **LangChain** montre que tu sais utiliser les outils standards du marché — LangChain/LangGraph est un mot-clé qui revient très souvent dans les offres GenAI/ML Engineer en France.

Avoir les deux dans un portfolio est un signal fort : compréhension des fondamentaux **et** maîtrise de l'écosystème professionnel.

## Ce qui est différent

| Aspect | `ollama-agent` (from scratch) | `langchain-ollama-agent` |
|---|---|---|
| Dépendances | `requests` uniquement | `langchain`, `langgraph`, `langchain-ollama` |
| Comment le modèle "décide" d'utiliser un outil | Le modèle écrit du texte au format `ACTION: outil(entrée)`, qu'on parse nous-même avec des regex | Le modèle utilise le **tool calling natif** (function calling) : il renvoie directement une structure JSON indiquant quel outil appeler et avec quels arguments |
| Boucle de l'agent | Codée à la main dans `agent/core.py` (`for step in range(...)`) | Fournie par LangGraph via `create_react_agent` |
| Définir un outil | Fonction Python + entrée dans un dictionnaire `REGISTRY` | Fonction Python décorée avec `@tool` (LangChain) |
| Écosystème disponible | Aucun — tout est à construire | Mémoire de conversation, retrievers RAG, traçage (LangSmith), intégrations tierces prêtes à l'emploi |
| Ce que ça démontre | Compréhension du fonctionnement interne d'un agent | Maîtrise des outils utilisés en entreprise |

## Ce qui est IDENTIQUE

- Le comportement final pour l'utilisateur (poser une question, obtenir une réponse, avec ou sans appel d'outil)
- Les outils proposés : `calculatrice`, `compte_mots`, `inverse_texte`, `recherche_web`
- L'architecture à plugins : dans les deux versions, ajouter un outil = créer un fichier dans `agent/tools/`, sans toucher au reste du code
- Le modèle utilisé : Ollama en local, gratuit, aucune clé API nécessaire

## Quand utiliser laquelle (en entretien)

- On te demande d'expliquer **comment fonctionne un agent** → montre la version from scratch, tu peux dérouler la boucle ligne par ligne
- On te demande si tu connais **LangChain/LangGraph** → montre la version LangChain, mentionne que tu as aussi la version from scratch pour prouver que tu ne l'utilises pas comme une boîte noire

## Installation (commune aux deux)

```bash
# Installer Ollama : https://ollama.com/download
ollama pull llama3.1     # doit supporter le tool calling pour la version LangChain
ollama serve
```

Puis suivre le README propre à chaque dépôt pour l'installation des dépendances Python.
