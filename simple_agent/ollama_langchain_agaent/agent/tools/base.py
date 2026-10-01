"""
Registre global des outils LangChain (BaseTool).

Pour ajouter un nouvel outil : crée un fichier .py dans ce dossier,
définis une fonction décorée avec @tool (LangChain), puis enregistre-la
avec @register. Elle est automatiquement chargée par __init__.py et
devient disponible pour l'agent, sans toucher au reste du code.
"""

from typing import List
from langchain_core.tools import BaseTool

TOOLS: List[BaseTool] = []


def register(tool: BaseTool) -> BaseTool:
    TOOLS.append(tool)
    return tool
