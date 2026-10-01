"""
Construction de l'agent tool-calling avec LangGraph + Ollama.

On utilise `create_react_agent` de LangGraph (l'approche recommandée par
LangChain depuis la dépréciation d'AgentExecutor) : il gère nativement le
tool calling structuré du modèle (pas de parsing regex ACTION/FINAL —
Ollama/le modèle renvoie directement des appels d'outils au format JSON).

Le modèle Ollama choisi doit supporter le tool calling (ex : llama3.1,
qwen2.5, mistral-nemo). Voir : https://ollama.com/search?c=tools
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from agent.tools import TOOLS

SYSTEM_PROMPT = (
    "Tu es un assistant utile. Utilise les outils disponibles quand ils "
    "t'aident à répondre plus précisément. Réponds toujours en français."
)


def build_agent(model : str = "llama3.1", temperature: float = 0.2, base_url: str = "http://localhost:11434"):
    llm = ChatOllama(model=model, temperature=temperature, base_url=base_url)
    return create_react_agent(llm, tools=TOOLS, prompt=SYSTEM_PROMPT)

def ask(agent, question: str):
    """Envoie une question à l'agent et retourne le texte de la réponse finale."""
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content