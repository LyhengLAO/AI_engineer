"""
Charge automatiquement tous les modules d'outils présents dans ce dossier,
afin que leurs décorateurs @register_tool s'exécutent et remplissent le
REGISTRY. C'est ce mécanisme qui permet de "scaler" simplement : ajouter
un fichier ici suffit, aucune autre modification n'est nécessaire.
"""

import importlib
import pkgutil
from pathlib import Path

from .base import REGISTRY, ToolSpec, register_tool

_package_dir = Path(__file__).parent

for _module_info in pkgutil.iter_modules([str(_package_dir)]):
    if _module_info.name in {"base"}:
        continue
    importlib.import_module(f"{__name__}.{_module_info.name}")

__all__ = ["REGISTRY", "ToolSpec", "register_tool"]
