"""Coeur de l'agent : construction du prompt, boucle ReAct, parsing."""

from pydoc import text
import re
from typing import Tuple, Optional

from .tools import REGISTRY
from .ollama_client import OllamaClient

ACTION_PATTERN = re.compile(r"ACTION:\s*(\w+)\((.*)\)", re.DOTALL)
FINAL_PATTERN = re.compile(r"FINAL:\s*(.*)", re.DOTALL)

def build_system_prompt() -> str:
    """
    Construit le prompt système pour l'agent.

    Returns:
        str: Le prompt système.
    """
    tools_desc = "\n".join(
        f"- {spec.name}(entrée) : {spec.description}" for spec in REGISTRY.values()
    )
    return f"""Tu es un agent IA qui résout des tâches étape par étape.

Tu disposes des outils suivants :
{tools_desc}

Règles STRICTES de format :
- Si tu as besoin d'un outil, réponds UNIQUEMENT avec : ACTION: nom_outil(entrée)
- Si tu as la réponse finale, réponds UNIQUEMENT avec : FINAL: ta réponse ici
- Ne mélange jamais ACTION et FINAL dans le même message.
- N'affiche que la ligne ACTION ou FINAL, sans texte autour.
"""

def parse_response(text: str) -> Tuple[str, str, Optional[str]]:
    """Retourne ('action', nom_outil, entrée) ou ('final', réponse, None)."""
    action_match = ACTION_PATTERN.search(text)
    if action_match:
        tool_name = action_match.group(1).strip()
        tool_input = action_match.group(2).strip().strip('"').strip("'")
        return "action", tool_name, tool_input

    final_match = FINAL_PATTERN.search(text)
    if final_match:
        return "final", final_match.group(1).strip(), None

    # Filet de sécurité si le modèle ne respecte pas le format
    return "final", text, None

class Agent:
    def __init__(
        self,
        model: str = "llama3.1",
        max_iterations: int = 6,
        verbose: bool = True,
    ):
        self.client = OllamaClient(model=model)
        self.max_iterations = max_iterations
        self.verbose = verbose

    def run(self, question: str) -> str:
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": question},
        ]

        for step in range(1, self.max_iterations + 1):
            raw_response = self.client.chat(messages)
            kind, payload, tool_input = parse_response(raw_response)

            if self.verbose:
                print(f"\n--- Étape {step} ---")
                print(f"Modèle : {raw_response}")

            if kind == "final":
                return payload

            tool_name = payload
            spec = REGISTRY.get(tool_name)
            if spec is None:
                observation = (
                    f"Erreur : outil '{tool_name}' inconnu. "
                    f"Outils disponibles : {list(REGISTRY.keys())}"
                )
            else:
                observation = spec.func(tool_input)

            if self.verbose:
                print(f"Observation : {observation}")

            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user", "content": f"Résultat de l'outil : {observation}"})

        return "Nombre maximum d'itérations atteint sans réponse finale."
