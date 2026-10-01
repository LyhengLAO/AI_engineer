import argparse

from agent import build_agent, ask


def main():
    parser = argparse.ArgumentParser(description="Agent tool-calling LangChain/LangGraph + Ollama (100% local)")
    parser.add_argument("--model", default="llama3.1", help="Modèle Ollama supportant le tool calling (défaut : llama3.1)")
    parser.add_argument("--question", "-q", help="Pose une question directement (mode non-interactif)")
    args = parser.parse_args()

    graph = build_agent(model=args.model)

    if args.question:
        print(ask(graph, args.question))
        return

    print("Agent LangChain/LangGraph (Ollama) — tape 'quit' pour quitter\n")
    while True:
        question = input("Toi : ")
        if question.lower() in {"quit", "exit"}:
            break
        print(f"\n=== Réponse ===\n{ask(graph, question)}\n")


if __name__ == "__main__":
    main()
