"""Quick end-to-end test of the AutoBrain agent."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent import AutoBrainAgent

agent = AutoBrainAgent()

result = agent.answer_query("How does the tire pressure monitoring system work?")

print("\n--- ANSWER ---")
print(result["answer"])

print("\n--- SOURCES ---")
print([f"Page {c['page']}" for c in result["retrieved"]])

print("\n--- FOLLOW-UPS ---")
for q in result["follow_ups"]:
    print(f"  > {q}")
