"""
Système de plugins pour les outils de l'agent.

Pour ajouter un nouvel outil, il suffit de créer un fichier .py dans ce
dossier (agent/tools/) et de décorer une fonction avec @register_tool.
Elle sera automatiquement détectée et disponible pour l'agent, sans
toucher au reste du code.
"""

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable[[str], str]


# Registre global : nom_outil -> ToolSpec
REGISTRY: Dict[str, ToolSpec] = {}


def register_tool(name: str, description: str):
    """Décorateur pour enregistrer une fonction comme outil de l'agent.

    Exemple :
        @register_tool("ma_calculatrice", "Additionne deux nombres, ex: ma_calculatrice(2,3)")
        def additionner(entree: str) -> str:
            a, b = entree.split(",")
            return str(int(a) + int(b))
    """
    def decorator(func: Callable[[str], str]):
        if name in REGISTRY:
            raise ValueError(f"Un outil nommé '{name}' est déjà enregistré.")
        REGISTRY[name] = ToolSpec(name=name, description=description, func=func)
        return func
    return decorator
