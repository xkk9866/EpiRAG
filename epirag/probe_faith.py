"""Print one FaithEval inconsistent row so the runner can use real fields."""

from datasets import load_dataset

data = load_dataset("Salesforce/FaithEval-inconsistent-v1.0", split="test")
print("n", len(data))
row = data[0]
print("keys", list(row.keys()))
for key, value in row.items():
    text = str(value)
    print(f"\n## {key} ({len(text)} chars)")
    print(text[:800])
