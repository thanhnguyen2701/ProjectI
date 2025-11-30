import json

FILE = r"C:\Users\FPTSHOP\2025.1\ProjectI\evaluation_llm_boolean.jsonl"

total = 0
true_count = 0

with open(FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        total += 1
        if item.get("llm_evaluation", {}).get("correct") is True:
            true_count += 1

percent = (true_count / total) * 100 if total > 0 else 0

print("Total:", total)
print("True:", true_count)
print("Accuracy:", f"{percent:.2f}%")
