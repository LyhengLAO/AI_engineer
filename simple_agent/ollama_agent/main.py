import argparse

from agent import Agent


def main():
    parser = argparse.ArgumentParser(description="Agent IA ReAct basé sur Ollama (100% local et gratuit)")
    parser.add_argument("--model", default="llama3.1", help="Nom du modèle Ollama à utiliser (défaut : llama3.1)")
    parser.add_argument("--question", "-q", help="Pose une question directement (mode non-interactif)")
    parser.add_argument("--quiet", action="store_true", help="N'affiche que la réponse finale")
    args = parser.parse_args()

    agent = Agent(model=args.model, verbose=not args.quiet)

    if args.question:
        print(agent.run(args.question))
        return

    print("Agent IA (Ollama) — tape 'quit' pour quitter\n")
    while True:
        question = input("Toi : ")
        if question.lower() in {"quit", "exit"}:
            break
        answer = agent.run(question)
        print(f"\n=== Réponse finale ===\n{answer}\n")


if __name__ == "__main__":
    main()
