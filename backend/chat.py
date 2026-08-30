"""
AutoBrain Lite — CLI Chatbot Interface

Runs a local console chat session with the AutoBrain agent.
Requires GGUF model and FAISS database to be ready.

Usage:
    python backend/chat.py
"""

import os
import sys

# Ensure backend directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent import AutoBrainAgent


def main():
    print("\n" + "=" * 60)
    print("AutoBrain Lite -- Local Console Chatbot")
    print("=" * 60)
    print("  Initializing the local RAG assistant. Please wait...")

    try:
        agent = AutoBrainAgent(
            model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
            index_dir="data"
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] Initialization failed: {e}")
        print("Please make sure you downloaded the model and ingested a PDF manual first.")
        sys.exit(1)

    print("\n[INFO] Assistant is ready! Ask any questions about your vehicle.")
    print("       Type 'exit' or 'quit' to end the chat.\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            
            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            # Answer query end-to-end
            result = agent.answer_query(query)

            print("\nAutoBrain:")
            print(result["answer"])
            print()

            # Display citations if available
            if result["retrieved"]:
                pages = sorted(list(set(c["page"] for c in result["retrieved"])))
                pages_str = ", ".join(f"Page {p}" for p in pages)
                print(f"[Sources: {pages_str}]")

            # Display follow-up questions
            if result["follow_ups"]:
                print("\nSuggested follow-ups:")
                for q in result["follow_ups"]:
                    print(f"  > {q}")
            
            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error occurred] {str(e)}\n")


if __name__ == "__main__":
    main()
