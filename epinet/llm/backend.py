"""Cached, batched LLM access.

All generation is greedy. Completions are cached on disk by a hash of the
backend name, the prompt and the token limit, so repeated prompts across
methods and conditions are computed once and every run is replayable.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from epinet.text import stable_hash


class Backend:
    name = "backend"

    def complete(self, prompts: list[str], max_tokens: int) -> list[str]:
        raise NotImplementedError

    def count(self, text: str) -> int:
        return int(round(len((text or "").split()) * 1.3))


class MockBackend(Backend):
    """Deterministic backend for tests: ``fn(prompt) -> completion``."""

    def __init__(self, fn: Callable[[str], str], name: str = "mock"):
        self.fn = fn
        self.name = name

    def complete(self, prompts, max_tokens):
        return [self.fn(p) for p in prompts]


class HFBackend(Backend):
    """Local transformers model, left-padded greedy batches."""

    def __init__(self, model: str, device: str = "cuda:0", batch_size: int = 8, load_in_4bit: bool = False,
                 max_input: int = 3072, load_in_8bit: bool = False, load_ao8: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model + ("@4bit" if load_in_4bit else "") + ("@int8" if load_in_8bit else "") + ("@ao8" if load_ao8 else "")
        self.device = device
        self.batch_size = int(os.environ.get("EPINET_BATCH", batch_size))
        self.max_input = max_input
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        kwargs = {"torch_dtype": torch.float16, "trust_remote_code": True}
        if load_ao8:
            # INT8 weight-only (torchao): every linear weight is stored as int8, activations bf16
            from torchao.quantization import Int8WeightOnlyConfig
            from transformers import TorchAoConfig

            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["quantization_config"] = TorchAoConfig(Int8WeightOnlyConfig())
            kwargs["device_map"] = {"": device}
        elif load_in_8bit:
            # integer weights (LLM.int8): every linear weight is stored as int8
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = {"": device}
        elif load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
            )
            kwargs["device_map"] = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if not (load_in_4bit or load_in_8bit or load_ao8):
            self.model.to(device)
        self.model.eval()
        self._torch = torch

    def _render(self, prompt: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: direct answers; ignored by templates without the switch
        )

    def complete(self, prompts, max_tokens):
        torch = self._torch
        out: list[str] = []
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        texts = {}
        batch = self.batch_size
        cursor = 0
        while cursor < len(order):
            idx = order[cursor : cursor + batch]
            try:
                enc = self.tok(
                    [self._render(prompts[i]) for i in idx],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input,
                ).to(self.device)
                with torch.inference_mode():
                    gen = self.model.generate(
                        **enc, max_new_tokens=max_tokens, do_sample=False,
                        pad_token_id=self.tok.pad_token_id,
                    )
                decoded = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i, text in zip(idx, decoded):
                    texts[i] = text.strip()
                cursor += batch
            except RuntimeError as exc:  # pragma: no cover - hardware dependent
                if "out of memory" not in str(exc).lower() or batch == 1:
                    raise
                torch.cuda.empty_cache()
                batch = max(1, batch // 2)
        for i in range(len(prompts)):
            out.append(texts[i])
            if (i + 1) % 500 == 0:
                print(f"generated {i + 1}/{len(prompts)}", flush=True)
        return out

    def count(self, text: str) -> int:
        return len(self.tok.encode(text or "", add_special_tokens=False))


class OpenAICompatBackend(Backend):
    """Any OpenAI-compatible chat endpoint (llama.cpp server, vLLM, Ollama)."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434/v1", workers: int = 8,
                 tokenizer: str | None = None, seed: int = 7, timeout: float = 600.0):
        import httpx

        self.name = f"{model}@{base_url}"
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.workers = workers
        self.seed = seed
        self.client = httpx.Client(timeout=timeout)
        self.tok = None
        if tokenizer:
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def _one(self, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": self.seed,
            "max_tokens": max_tokens,
        }
        for attempt in range(5):
            try:
                resp = self.client.post(self.url, json=payload)
                resp.raise_for_status()
                return (resp.json()["choices"][0]["message"]["content"] or "").strip()
            except Exception:  # pragma: no cover - network dependent
                if attempt == 4:
                    raise
        return ""

    def complete(self, prompts, max_tokens):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda p: self._one(p, max_tokens), prompts))

    def count(self, text: str) -> int:
        if self.tok is None:
            return super().count(text)
        return len(self.tok.encode(text or "", add_special_tokens=False))


class CachedLLM:
    def __init__(self, backend: Backend, cache_path: str | None = None):
        self.backend = backend
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self.calls = 0
        self.lock = threading.Lock()
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.cache[row["key"]] = row["text"]

    @property
    def name(self) -> str:
        return self.backend.name

    def key(self, prompt: str, max_tokens: int) -> str:
        return stable_hash(f"{self.backend.name}\n<max:{max_tokens}>\n{prompt}")

    def generate(self, prompts: list[str], max_tokens: int = 48) -> list[str]:
        results: list[str | None] = [None] * len(prompts)
        missing: dict[str, list[int]] = {}
        for i, prompt in enumerate(prompts):
            k = self.key(prompt, max_tokens)
            if k in self.cache:
                results[i] = self.cache[k]
            else:
                missing.setdefault(k, []).append(i)
        if missing:
            keys = list(missing)
            todo = [prompts[missing[k][0]] for k in keys]
            texts = self.backend.complete(todo, max_tokens)
            self.calls += len(todo)
            lines = []
            for k, text in zip(keys, texts):
                self.cache[k] = text
                for i in missing[k]:
                    results[i] = text
                lines.append(json.dumps({"key": k, "text": text}, ensure_ascii=False))
            if self.cache_path:
                with self.lock:
                    os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
                    with open(self.cache_path, "a", encoding="utf-8") as handle:
                        handle.write("\n".join(lines) + "\n")
        return [r or "" for r in results]

    def count(self, text: str) -> int:
        return self.backend.count(text)


def make_llm(spec: str, cache_dir: str = "experiments/cache") -> CachedLLM:
    """``hf:<model>[@4bit]`` or ``http:<model>@<base_url>``."""
    if spec.startswith("hf:"):
        model = spec[3:]
        int8 = model.endswith("@int8")
        four = model.endswith("@4bit")
        ao8 = model.endswith("@ao8")
        model = model.replace("@4bit", "").replace("@int8", "").replace("@ao8", "")
        backend = HFBackend(model, device=os.environ.get("EPINET_DEVICE", "cuda:0"),
                            load_in_4bit=four, load_in_8bit=int8, load_ao8=ao8)
    elif spec.startswith("http:"):
        body = spec[5:]
        model, _, base = body.partition("@")
        backend = OpenAICompatBackend(
            model, base_url=base or "http://127.0.0.1:11434/v1",
            workers=int(os.environ.get("EPINET_WORKERS", "8")),
            tokenizer=os.environ.get("EPINET_TOKENIZER"),
        )
    else:
        raise ValueError(spec)
    safe = stable_hash(backend.name)[:10]
    return CachedLLM(backend, os.path.join(cache_dir, f"gen_{safe}.jsonl"))
