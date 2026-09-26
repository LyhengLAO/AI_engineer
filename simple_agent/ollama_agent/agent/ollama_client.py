"""Client minimal pour l'API locale d'Ollama (/api/chat)."""

import requests
from typing import List, Dict

class OllamaClient:
    def __init__(self, model : str = "llama3.1",
                 base_url : str = "http://localhost:11434",
                 temperature : float = 0.2,
                 timeout : int = 120
                 ):
        self.model = model
        self.base_url = f"{base_url}/api/chat"
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages : List[Dict[str, str]]) -> str:
        """
        Envoie une requête de chat à l'API locale d'Ollama.

        Args:
            messages (List[Dict[str, str]]): Liste de messages à envoyer à l'API.

        Returns:
            str: La réponse du modèle.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }
        response = requests.post(self.base_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()