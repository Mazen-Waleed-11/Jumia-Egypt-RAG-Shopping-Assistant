"""End-to-end multi-turn test of rag.py without interactive input."""
import sys
sys.path.insert(0, ".")
from rag import (
    load_resources, handle_turn, Session,
    detect_subcategory, parse_price_filter, resolve_reference,
    is_followup,
)

print("=" * 70)
print("UNIT CHECKS")
print("=" * 70)

print("\n[A] sub-category detection")
for q in [
    "Samsung phone under 10k",
    "what about cheaper ones?",
    "I need a gaming laptop",
    "wireless earbuds",
    "tell me about a power bank",
]:
    print(f"  {q!r:55} -> {detect_subcategory(q)!r}")

print("\n[B] price parsing")
for q in [
    "phone under 10k",
    "between 5000 and 9000 EGP",
    "around 8000",
    "above 15000 LE",
    "Samsung Galaxy A07",
]:
    print(f"  {q!r:55} -> {parse_price_filter(q)}")

print("\n[C] reference resolution")
fake = [{"Name": "Phone A", "Price": 1000},
        {"Name": "Phone B", "Price": 2000},
        {"Name": "Phone C", "Price": 3000}]
for q in [
    "tell me about #2",
    "the second one please",
    "what is the price of the first one",
    "how much is the last?",
    "tell me about it",
    "option 3",
]:
    r = resolve_reference(q, fake)
    print(f"  {q!r:55} -> {r['Name'] if r else None}")

print("\n[D] follow-up detection")
for q in [
    "what about cheaper ones?",
    "and a smaller one?",
    "any other suggestions?",
    "I want a phone under 10k",
]:
    print(f"  {q!r:55} -> {is_followup(q)}")

print()
print("=" * 70)
print("END-TO-END (Ollama)")
print("=" * 70)

res = load_resources()
sess = Session()

queries = [
    "I want a Samsung phone under 10000 EGP",
    "what about cheaper ones?",
    "tell me about #2",
]

for q in queries:
    print()
    print("-" * 70)
    print(f"USER: {q}")
    print("-" * 70)
    answer = handle_turn(q, res, sess)
    sess.remember(q, answer)
