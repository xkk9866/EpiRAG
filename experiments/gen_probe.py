"""One short generation to see whether the 7B can decode in the free GPU margin."""

import sys

from epirag.engine import Generator

device = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
print("loading", flush=True)
generator = Generator(
    "Qwen/Qwen2.5-7B-Instruct",
    device,
    "experiments/cache/generations.jsonl",
    batch_size=1,
)
print("generating", flush=True)
text = generator.generate(["Reply with the single word hi."], max_new_tokens=8)
print(text, flush=True)
