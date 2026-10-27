"""Check which hypothesis template the NLI model actually separates."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = "cuda:1" if torch.cuda.is_available() else "cpu"
name = "cross-encoder/nli-deberta-v3-base"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
model.eval()
premise = (
    "The M104 155 mm projectile was designed to carry about 11.7 pounds of sulfur mustard, "
    "a prototypical blister agent."
)
contradict = (
    "The M104 155 mm projectile was designed to carry about 11.7 pounds of white phosphorus."
)
templates = {
    "answer_is": "The answer is sulfur mustard.",
    "span": "sulfur mustard",
    "filled": "The projectile was designed to carry sulfur mustard.",
    "q": "What substance did it carry? sulfur mustard",
    "wrong": "The answer is white phosphorus.",
}
pairs = [("good", premise), ("bad", contradict)]
hypos = list(templates.items())
premises = []
hypotheses = []
labels = []
for tag, text in pairs:
    for hname, hypo in hypos:
        premises.append(text)
        hypotheses.append(hypo)
        labels.append(f"{tag}|{hname}")
enc = tok(premises, hypotheses, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
with torch.inference_mode():
    prob = torch.softmax(model(**enc).logits, dim=-1).cpu().tolist()
for label, p in zip(labels, prob):
    print(f"{label:22} con={p[0]:.3f} ent={p[1]:.3f} neu={p[2]:.3f}")
