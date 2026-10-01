"""
Charge automatiquement tous les modules d'outils de ce dossier au démarrage,
afin que leurs décorateurs @register s'exécutent et remplissent TOOLS.
C'est ce qui permet de scaler simplement : ajouter un fichier ici suffit.
"""

import importlib
import pkgutil
from pathlib import Path

from .base import TOOLS, register

_package_dir = Path(__file__).parent

for _module_info in pkgutil.iter_modules([str(_package_dir)]):
    if _module_info.name == "base":
        continue
    importlib.import_module(f"{__name__}.{_module_info.name}")

__all__ = ["TOOLS", "register"]
