"""Count FaithEval documents and conflicting answers. No model."""

from collections import Counter

from datasets import load_dataset

data = load_dataset("Salesforce/FaithEval-inconsistent-v1.0", split="test")
docs = Counter()
n_ans = Counter()
for row in data:
    parts = [part.strip() for part in str(row["context"]).split("Document:") if part.strip()]
    docs[len(parts)] += 1
    n_ans[len(row["answers"])] += 1
print("documents", dict(sorted(docs.items())))
print("n_answers", dict(sorted(n_ans.items())))
print("--- sample tail ---")
print(str(data[0]["context"])[-500:])
