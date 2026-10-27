"""Readers, NLI scoring, and epistemic coordination."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass

import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from rank_bm25 import BM25Okapi
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification

from epirag.data import Passage, tokenize
from epirag.metrics import token_f1


REFERENCE_YEAR = 2024
CONTRA_LAMBDA = 1.0
RELIABILITY_ALPHA = 0.5
MARGIN_THRESHOLD = 0.10
SUPPORT_FLOOR = 0.30
BUNDLE_K = 4
NLI_MAX_LEN = 256
MAX_CLAIM_TOKENS = 8
COVER_GAP = 0.45
SUPPORT_ENT = 0.40
RIVAL_ENT = 0.40
QUESTION_STOP = {
    "what", "which", "who", "where", "when", "whose", "whom", "that", "this",
    "from", "with", "have", "been", "were", "was", "are", "the", "and", "for",
    "how", "did", "does", "his", "her", "their", "about", "into", "than", "then",
    "both", "same", "more", "less", "also", "between", "first", "after", "before",
    "during", "while", "over", "under", "there", "these", "those", "your",
}


def load_causal_lm(model_name: str, device: str):
    """Place fp16 weights on one GPU without the from_pretrained map.

    On this machine the stock loader faults while materializing the 7B
    checkpoint. Reading the safetensors shards and assigning them onto an
    empty module keeps the same fp16 weights.
    """
    root = model_name if os.path.isdir(model_name) else snapshot_download(model_name)
    config = AutoConfig.from_pretrained(root, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    state: dict[str, torch.Tensor] = {}
    for path in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                state[key] = handle.get_tensor(key).to(device=device, dtype=torch.float16)

    def place(module, prefix: str):
        for name in list(module._parameters):
            key = f"{prefix}.{name}" if prefix else name
            if key in state:
                module._parameters[name] = torch.nn.Parameter(state.pop(key), requires_grad=False)
        for name in list(module._buffers):
            key = f"{prefix}.{name}" if prefix else name
            if key in state:
                module._buffers[name] = state.pop(key)
        for name, child in module.named_children():
            place(child, f"{prefix}.{name}" if prefix else name)

    place(model, "")
    if state:
        raise RuntimeError(f"unassigned weights: {sorted(state)[:8]}")
    model.tie_weights()
    model.eval()
    import gc
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def recency_prior(year: int) -> float:
    age = max(0, REFERENCE_YEAR - int(year))
    return float(math.exp(-0.12 * age))


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    shifted = [math.exp(v - peak) for v in values]
    total = sum(shifted) or 1.0
    return [v / total for v in shifted]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    kept = [p.strip() for p in parts if len(p.strip()) > 24]
    if not kept and text:
        kept = [text.strip()[:360]]
    return kept[:3]


def parse_answer(raw: str) -> tuple[str, str]:
    """Recover a short span, including an answer field cut off by the token limit."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    answer_match = re.search(r'"answer"\s*:\s*"([^"\n]*)', text)
    if answer_match and answer_match.group(1).strip():
        quote_match = re.search(r'"quote"\s*:\s*"([^"\n]*)', text)
        quote = quote_match.group(1).strip() if quote_match else ""
        return answer_match.group(1).strip()[:80], quote
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            answer = str(obj.get("answer", "")).strip()
            if answer:
                return answer[:80], str(obj.get("quote", "")).strip()
        except json.JSONDecodeError:
            pass
    line = text.splitlines()[0].strip() if text else ""
    line = re.sub(r"^(answer\s*:\s*)", "", line, flags=re.IGNORECASE).strip().strip('"')
    return line[:80], ""


def evidence_prompt(question: str, passages: list[Passage], instruction: str = "") -> str:
    blocks = []
    for i, passage in enumerate(passages, start=1):
        blocks.append(passage.prompt_block(i))
    evidence = "\n".join(blocks) if blocks else "[1] No evidence was retrieved."
    extra = f"\n{instruction}" if instruction else ""
    return (
        "Answer the question using only the evidence. "
        "Reply with the short answer span only. "
        "Do not explain, and do not use JSON.\n\n"
        f"Question: {question}\n"
        f"Evidence:\n{evidence}"
        f"{extra}"
    )


def revision_prompt(question: str, passages: list[Passage], local_answers: list[str]) -> str:
    """Ask the reader to revise agent proposals instead of answering from scratch.

    A unanimous proposal is the default. The reader replaces it only when the
    bundle states a different answer. Disagreement leaves the choice to the
    evidence, including a span that no agent proposed.
    """
    blocks = []
    for i, passage in enumerate(passages, start=1):
        blocks.append(passage.prompt_block(i))
    evidence = "\n".join(blocks) if blocks else "[1] No evidence was retrieved."
    proposed = [answer.strip() for answer in local_answers if answer and answer.strip()]
    folded = [re.sub(r"\s+", " ", answer.lower()) for answer in proposed]
    unique = []
    seen = set()
    for answer, key in zip(proposed, folded):
        if key in seen:
            continue
        seen.add(key)
        unique.append(answer)
    if len(folded) >= 2 and len(set(folded)) == 1:
        proposal = f"All agents proposed the same answer: {proposed[0]}."
        rule = (
            "Keep that answer when the evidence supports it. "
            "Replace it only when a sentence states a different answer."
        )
    else:
        proposal = "Agents proposed: " + " | ".join(unique) + "."
        rule = (
            "The agents disagree. Choose the short answer the evidence supports. "
            "You may use a span from the evidence that none of them proposed."
        )
    polar = ""
    if is_polar_question(question):
        polar = "This is a yes/no question. Answer yes or no.\n"
    return (
        "Answer the question using only the evidence. "
        "Reply with the short answer span only. "
        "Do not explain, and do not use JSON.\n"
        f"{proposal} {rule}\n"
        f"{polar}"
        f"Question: {question}\n"
        f"Evidence:\n{evidence}"
    )


def debate_prompt(
    question: str, passages: list[Passage], other_answers: list[str], instruction: str = ""
) -> str:
    base = evidence_prompt(question, passages, instruction)
    joined = "; ".join(a for a in other_answers if a) or "none"
    return (
        base
        + "\nOther agents answered: "
        + joined
        + "\nRevise your own answer only if their answers are supported by your evidence."
    )


class Generator:
    def __init__(self, model_name: str, device: str, cache_path: str, batch_size: int = 4):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        print("loading generator weights", flush=True)
        self.model = load_causal_lm(model_name, device)
        # The generation cache is large. Loading it before the weights
        # fragments the process heap and the 7B mapping then faults.
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    self.cache[row["key"]] = row["text"]
        except FileNotFoundError:
            pass
        print(f"generator cache entries {len(self.cache)}", flush=True)

    def _render(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompts: list[str], max_new_tokens: int = 64) -> list[str]:
        results = [""] * len(prompts)
        missing: list[tuple[int, str, str]] = []
        for idx, prompt in enumerate(prompts):
            key_text = self.model_name + "\n" + prompt
            if max_new_tokens != 64:
                key_text += f"\n<max:{max_new_tokens}>"
            key = stable_hash(key_text)
            if key in self.cache:
                results[idx] = self.cache[key]
            else:
                missing.append((idx, key, prompt))
        cursor = 0
        batch = self.batch_size
        while cursor < len(missing):
            chunk = missing[cursor : cursor + batch]
            try:
                texts = [self._render(item[2]) for item in chunk]
                encoded = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(self.device)
                with torch.inference_mode():
                    output = self.model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                new_tokens = output[:, encoded["input_ids"].shape[1] :]
                decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                with open(self.cache_path, "a", encoding="utf-8") as handle:
                    for (idx, key, _prompt), text in zip(chunk, decoded):
                        text = text.strip()
                        self.cache[key] = text
                        results[idx] = text
                        handle.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
                cursor += batch
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or batch == 1:
                    raise
                torch.cuda.empty_cache()
                batch = max(1, batch // 2)
                self.batch_size = batch
        return results

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text or "", add_special_tokens=False))


class NLI:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.cache: dict[tuple[str, str], tuple[float, float, float]] = {}

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[tuple[float, float, float]]:
        results: list[tuple[float, float, float] | None] = [None] * len(pairs)
        missing = []
        for idx, (premise, hypothesis) in enumerate(pairs):
            key = (stable_hash(premise), stable_hash(hypothesis))
            if key in self.cache:
                results[idx] = self.cache[key]
            else:
                missing.append((idx, key, premise, hypothesis))
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            encoded = self.tokenizer(
                [item[2] for item in chunk],
                [item[3] for item in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=NLI_MAX_LEN,
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            for item, prob in zip(chunk, probs):
                triple = (float(prob[0]), float(prob[1]), float(prob[2]))
                self.cache[item[1]] = triple
                results[item[0]] = triple
        return [item if item is not None else (0.0, 0.0, 1.0) for item in results]


@dataclass
class Snippet:
    text: str
    passage: Passage
    rho: float


def _rank_sentences(question: str, passages: list[Passage]) -> list[Snippet]:
    snippets: list[Snippet] = []
    for passage in passages:
        for sentence in split_sentences(passage.text):
            snippets.append(Snippet(text=sentence, passage=passage, rho=0.0))
    if not snippets:
        return []
    corpus = [tokenize(s.text) or ["empty"] for s in snippets]
    ranker = BM25Okapi(corpus)
    scores = ranker.get_scores(tokenize(question) or ["empty"])
    raw = [float(s) for s in scores]
    low, high = min(raw), max(raw)
    span = high - low
    for snippet, score in zip(snippets, raw):
        snippet.rho = 0.2 + 0.8 * ((score - low) / span) if span > 1e-8 else 1.0
    snippets.sort(key=lambda s: s.rho, reverse=True)
    # Keep the two most relevant sentences from each passage, then the global top.
    per_passage: dict[str, int] = {}
    chosen = []
    for snippet in snippets:
        count = per_passage.get(snippet.passage.pid, 0)
        if count >= 2:
            continue
        per_passage[snippet.passage.pid] = count + 1
        chosen.append(snippet)
    return chosen


def _dedup_passages(passages: list[Passage]) -> list[Passage]:
    seen = set()
    ordered = []
    for passage in passages:
        if passage.pid in seen:
            continue
        seen.add(passage.pid)
        ordered.append(passage)
    return ordered


def _merge_similar_claims(claims: list[str]) -> list[str]:
    """Collapse near-duplicate answer strings so they do not look like a conflict."""
    parent = list(range(len(claims)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            if token_f1(claims[i], claims[j]) >= 0.5:
                parent[find(j)] = find(i)
    groups: dict[int, list[str]] = {}
    for index, claim in enumerate(claims):
        groups.setdefault(find(index), []).append(claim)
    merged = []
    for group in groups.values():
        merged.append(min(group, key=lambda text: (len(text), text)))
    return merged


def _question_terms(question: str, claim: str) -> list[str]:
    blocked = set(tokenize(claim)) | QUESTION_STOP
    seen = []
    for token in tokenize(question):
        if len(token) <= 3 or token in blocked or token in seen:
            continue
        seen.append(token)
    return seen


def _is_rival_option(question: str, left: str, right: str) -> bool:
    text = question.lower()
    return left.lower() in text and right.lower() in text and left.lower() != right.lower()


def _short_claims(claims: list[str]) -> list[str]:
    short = [claim for claim in claims if len(tokenize(claim)) <= MAX_CLAIM_TOKENS]
    if short:
        return short
    if not claims:
        return ["unknown"]
    shortest = min(claims, key=lambda text: (len(tokenize(text)), len(text)))
    tokens = tokenize(shortest)[:MAX_CLAIM_TOKENS]
    return [" ".join(tokens)] if tokens else [shortest[:40]]


def _stated_in(sentence: str, claim: str) -> bool:
    toks = tokenize(claim)
    if not toks:
        return False
    sent = set(tokenize(sentence))
    return sum(token in sent for token in toks) / len(toks) >= 0.5


def _proposer_sources(local_answers: list[str], agent_passages: list[list[Passage]], claims: list[str]):
    owners = {claim: set() for claim in claims}
    for answer, passages in zip(local_answers, agent_passages):
        if not answer or not str(answer).strip():
            continue
        best = None
        best_f1 = 0.0
        for claim in claims:
            score = token_f1(answer, claim)
            if score > best_f1:
                best = claim
                best_f1 = score
        if best is None or best_f1 < 0.5:
            continue
        for passage in passages:
            if passage.source_id == "oracle":
                continue
            owners[best].add(passage.source_id)
    return owners


def _external_hits(snippets: list[Snippet], claims: list[str], owners: dict, table: dict):
    found = {}
    self_entail = {}
    for claim in claims:
        blocked = owners.get(claim, set())
        best = None
        own = 0.0
        for snippet in snippets:
            contra, entail, _neutral = table.get((snippet.text, claim), (0.0, 0.0, 1.0))
            if snippet.passage.source_id in blocked:
                own = max(own, entail)
                continue
            if entail >= SUPPORT_ENT and contra < 0.55 and _stated_in(snippet.text, claim):
                if best is None or entail > best[0]:
                    best = (entail, snippet)
        found[claim] = best
        self_entail[claim] = own
    return found, self_entail


def _question_lookup(question: str, archive: list[Passage], held: set[str]):
    pool = [p for p in archive if str(p.pid).startswith("p") and p.pid not in held]
    if not pool:
        return None
    corpus = [tokenize(p.title + " " + p.text) or ["empty"] for p in pool]
    ranker = BM25Okapi(corpus)
    scores = ranker.get_scores(tokenize(question) or ["empty"])
    return pool[max(range(len(pool)), key=lambda i: float(scores[i]))]


def _bridge_hits(snippets: list[Snippet], claims: list[str], owners: dict, table: dict):
    """A conclusion is a claim stated by its own proposer and anchored on another.

    The anchor is a different claim, proposed by a disjoint set of sources, whose
    tokens occur in that same sentence. This is the multi-hop case: the final
    span is not copied into a second view, but the second view names the first
    view's entity.
    """
    found = {}
    for claim in claims:
        own_sources = owners.get(claim, set())
        best = None
        for snippet in snippets:
            if snippet.passage.source_id not in own_sources:
                continue
            contra, entail, _neutral = table.get((snippet.text, claim), (0.0, 0.0, 1.0))
            if entail < SUPPORT_ENT or contra >= 0.55 or not _stated_in(snippet.text, claim):
                continue
            anchored = False
            for other in claims:
                if other == claim:
                    continue
                other_sources = owners.get(other, set())
                if not other_sources or other_sources & own_sources:
                    continue
                if _stated_in(snippet.text, other):
                    anchored = True
                    break
            if not anchored:
                continue
            if best is None or entail > best[0]:
                best = (entail, snippet)
        found[claim] = best
    return found


def _graph_bundle(snippets: list[Snippet], claims: list[str], table: dict) -> list[Snippet]:
    ranked = []
    for snippet in snippets:
        best_entail = -1.0
        best_contra = 1.0
        for claim in claims:
            contra, entail, _neutral = table.get((snippet.text, claim), (0.0, 0.0, 1.0))
            if entail > best_entail:
                best_entail = entail
                best_contra = contra
        if best_contra > 0.55 and best_entail < 0.25:
            continue
        ranked.append((best_entail, snippet))
    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = [snippet for _score, snippet in ranked[:BUNDLE_K]]
    return chosen or snippets[:BUNDLE_K]


def _ranked_pages(question: str, passages: list[Passage], k: int = 6) -> list[Passage]:
    if len(passages) <= 1:
        return list(passages)[:k]
    corpus = [tokenize(p.title + " " + p.text) or ["empty"] for p in passages]
    ranker = BM25Okapi(corpus)
    scores = ranker.get_scores(tokenize(question) or ["empty"])
    order = sorted(range(len(passages)), key=lambda i: float(scores[i]), reverse=True)
    return [passages[i] for i in order[:k]]


def _page_pieces(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    kept = [part.strip() for part in parts if part.strip()]
    if not kept and (text or "").strip():
        return [text.strip()]
    return kept


def _sentence_tag(sentence: str, claims: list[str], table: dict) -> str:
    """Label one sentence with the frozen support and contradiction cutoffs."""
    best_support = None
    best_entail = -1.0
    best_contra_claim = None
    best_contra = -1.0
    for claim in claims:
        contra, entail, _neutral = table.get((sentence, claim), (0.0, 0.0, 1.0))
        if entail >= SUPPORT_ENT and contra < 0.55 and entail > best_entail:
            best_entail = entail
            best_support = claim
        if contra > 0.55 and entail < 0.25 and contra > best_contra:
            best_contra = contra
            best_contra_claim = claim
    if best_contra_claim and (best_support is None or best_contra > best_entail):
        return f"CONTRADICTS {best_contra_claim}:"
    if best_support:
        return f"SUPPORTS {best_support}:"
    return "BACKGROUND:"


def bridge_entity_prompt(question: str, passages: list[Passage]) -> str:
    """Ask for the entity the question needs before the final span."""
    blocks = [passage.prompt_block(i) for i, passage in enumerate(passages, start=1)]
    evidence = "\n".join(blocks) if blocks else "[1] No evidence was retrieved."
    return (
        "Name the one entity this question depends on before the final answer. "
        "Copy that entity from the evidence. "
        "Reply with the entity only. Do not explain, and do not use JSON.\n\n"
        f"Question: {question}\n"
        f"Evidence:\n{evidence}"
    )


def chain_answer_prompt(question: str, passages: list[Passage], bridge: str) -> str:
    """Second read: the full page pool, conditioned on the bridge entity."""
    blocks = [passage.prompt_block(i) for i, passage in enumerate(passages, start=1)]
    evidence = "\n".join(blocks) if blocks else "[1] No evidence was retrieved."
    return (
        "Answer the question using only the evidence. "
        f"Use this bridge entity from the evidence: {bridge}. "
        "Reply with the short answer span only. "
        "Do not explain, and do not use JSON.\n\n"
        f"Question: {question}\n"
        f"Evidence:\n{evidence}"
    )


def marked_prompt(question: str, passages: list[Passage], tags: dict[str, str]) -> str:
    """Full pages, with each sentence tagged. Contradictions stay visible."""
    blocks = []
    for index, passage in enumerate(passages, start=1):
        lines = [f"[{index}] {passage.title} (source={passage.source_id}, year={passage.year})"]
        for piece in _page_pieces(passage.text):
            lines.append(f"{tags.get(piece, 'BACKGROUND:')} {piece}")
        blocks.append("\n".join(lines))
    evidence = "\n".join(blocks) if blocks else "[1] No evidence was retrieved."
    return (
        "Answer the question using only the evidence. "
        "A sentence marked SUPPORTS states that claim. "
        "A sentence marked CONTRADICTS conflicts with that claim; "
        "do not copy an answer that is stated only by a CONTRADICTS sentence "
        "when another sentence supports a different claim. "
        "A BACKGROUND sentence can still hold a bridge fact. "
        "Reply with the short answer span only. "
        "Do not explain, and do not use JSON.\n\n"
        f"Question: {question}\n"
        f"Evidence:\n{evidence}"
    )


def _cross_source_contradiction(snippets, claims, owners, table, hat: str) -> bool:
    """Another view's claim-bearing page contradicts the leading claim.

    The cutoffs are the frozen contradiction filter: contradiction above 0.55
    and entailment below 0.25. The page must belong to a proposer of a
    different claim, so a source cannot veto its own leading string.
    """
    for snippet in snippets:
        contra, entail, _neutral = table.get((snippet.text, hat), (0.0, 0.0, 1.0))
        if contra <= 0.55 or entail >= 0.25:
            continue
        source = snippet.passage.source_id
        if source in owners.get(hat, set()):
            continue
        for other in claims:
            if other != hat and source in owners.get(other, set()):
                return True
    return False


def coordinate(
    question: str,
    archive: list[Passage],
    agent_passages: list[list[Passage]],
    local_answers: list[str],
    nli: NLI,
    use_reliability: bool = True,
    use_contradiction: bool = True,
    allow_retrieval: bool = True,
    mode: str = "flat",
    allow_conflict: bool = False,
) -> dict:
    union = _dedup_passages([p for group in agent_passages for p in group])
    snippets = _rank_sentences(question, union)
    claims = []
    seen_claims = set()
    for answer in local_answers:
        key = re.sub(r"\s+", " ", (answer or "").strip().lower())
        if not key or key in seen_claims:
            continue
        seen_claims.add(key)
        claims.append(answer.strip())
    if not claims:
        claims = ["unknown"]
    claims = _merge_similar_claims(claims)
    if mode == "typed":
        claims = _short_claims(claims)
    if not snippets:
        snippets = [
            Snippet(text=p.text[:240], passage=p, rho=1.0)
            for p in union[:BUNDLE_K]
        ]

    def relations(snippet_list: list[Snippet], claim_list: list[str]):
        pairs = []
        for snippet in snippet_list:
            for claim in claim_list:
                pairs.append((snippet.text, claim_hypothesis(question, claim)))
        scored = nli.score(pairs)
        table = {}
        cursor = 0
        for snippet in snippet_list:
            for claim in claim_list:
                table[(snippet.text, claim)] = scored[cursor]
                cursor += 1
        return table

    table = relations(snippets, claims)
    reported_mode = mode
    decision_label = "reader"
    if mode == "gate":
        use_reliability = False
        owners = _proposer_sources(local_answers, agent_passages, claims)
        prelim = []
        for claim in claims:
            total = 0.0
            for snippet in snippets:
                contra, entail, _neutral = table.get((snippet.text, claim), (0.0, 0.0, 1.0))
                total += snippet.rho * (entail - CONTRA_LAMBDA * contra)
            prelim.append(total)
        rank = sorted(range(len(claims)), key=lambda i: prelim[i], reverse=True)
        hat = claims[rank[0]]
        beliefs = softmax(prelim)
        margin = 1.0 if len(rank) == 1 else beliefs[rank[0]] - beliefs[rank[1]]
        if _cross_source_contradiction(snippets, claims, owners, table, hat):
            if allow_conflict:
                return {
                    "claims": claims,
                    "beliefs": beliefs,
                    "hat": hat,
                    "bundle": snippets[:BUNDLE_K],
                    "reliability": {snippet.passage.source_id: 1.0 for snippet in snippets},
                    "triggered": False,
                    "extra_passages": [],
                    "margin": margin,
                    "coverage_gap": None,
                    "mode": "gate",
                    "structural": "conflict",
                    "decision": "conflict",
                    "nli_pairs": len(table),
                }
            mode = "cover"
            decision_label = "witness"
        else:
            pages = _ranked_pages(question, union, 6)
            chosen_ids = {page.pid for page in pages}
            support_snips = [snippet for snippet in snippets if snippet.passage.pid in chosen_ids]
            if support_snips:
                support_mass = sum(
                    table.get((snippet.text, hat), (0.0, 0.0, 1.0))[1] for snippet in support_snips
                ) / len(support_snips)
            else:
                support_mass = 0.0
            triggered = False
            extra_passages = []
            if allow_retrieval and (margin < MARGIN_THRESHOLD or support_mass < SUPPORT_FLOOR):
                triggered = True
                found_page = _question_lookup(question, archive, chosen_ids)
                if found_page is not None:
                    extra_passages = [found_page]
                    pages = list(pages) + [found_page]
                    extra_snips = _rank_sentences(question, [found_page])[:2]
                    if extra_snips:
                        table.update(relations(extra_snips, claims))
            bundle = []
            seen_pages = set()
            for page in pages:
                if page.pid in seen_pages:
                    continue
                seen_pages.add(page.pid)
                text = next((snippet.text for snippet in snippets if snippet.passage.pid == page.pid), page.text[:240])
                bundle.append(Snippet(text=text, passage=page, rho=1.0))
            return {
                "claims": claims,
                "beliefs": beliefs,
                "hat": hat,
                "bundle": bundle,
                "reliability": {page.source_id: 1.0 for page in pages},
                "triggered": triggered,
                "extra_passages": extra_passages,
                "margin": margin,
                "coverage_gap": None,
                "mode": "gate",
                "structural": "",
                "decision": "pool",
                "nli_pairs": len(table),
            }
    if mode == "mark":
        # The reader still sees the BM25 pool. A cross-source contradiction
        # does not swap in a four-sentence bundle. It labels every sentence
        # so a conflicting copy stays on the page but cannot be copied blindly.
        use_reliability = False
        owners = _proposer_sources(local_answers, agent_passages, claims)
        prelim = []
        for claim in claims:
            total = 0.0
            for snippet in snippets:
                contra, entail, _neutral = table.get((snippet.text, claim), (0.0, 0.0, 1.0))
                total += snippet.rho * (entail - CONTRA_LAMBDA * contra)
            prelim.append(total)
        rank = sorted(range(len(claims)), key=lambda i: prelim[i], reverse=True)
        hat = claims[rank[0]]
        beliefs = softmax(prelim)
        margin = 1.0 if len(rank) == 1 else beliefs[rank[0]] - beliefs[rank[1]]
        pages = _ranked_pages(question, union, 6)
        chosen_ids = {page.pid for page in pages}
        support_snips = [snippet for snippet in snippets if snippet.passage.pid in chosen_ids]
        if support_snips:
            support_mass = sum(
                table.get((snippet.text, hat), (0.0, 0.0, 1.0))[1] for snippet in support_snips
            ) / len(support_snips)
        else:
            support_mass = 0.0
        triggered = False
        extra_passages = []
        if allow_retrieval and (margin < MARGIN_THRESHOLD or support_mass < SUPPORT_FLOOR):
            triggered = True
            found_page = _question_lookup(question, archive, chosen_ids)
            if found_page is not None:
                extra_passages = [found_page]
                pages = list(pages) + [found_page]
        pieces = [
            Snippet(text=piece, passage=page, rho=1.0)
            for page in pages
            for piece in _page_pieces(page.text)
        ]
        missing = [
            piece
            for piece in pieces
            if any((piece.text, claim) not in table for claim in claims)
        ]
        if missing:
            table.update(relations(missing, claims))
        cross = _cross_source_contradiction(pieces, claims, owners, table, hat)
        tags: dict[str, str] = {}
        if cross:
            for piece in pieces:
                tags[piece.text] = _sentence_tag(piece.text, claims, table)
        bundle = []
        seen_pages: set[str] = set()
        for page in pages:
            if page.pid in seen_pages:
                continue
            seen_pages.add(page.pid)
            bundle.append(Snippet(text=page.text[:240], passage=page, rho=1.0))
        return {
            "claims": claims,
            "beliefs": beliefs,
            "hat": hat,
            "bundle": bundle,
            "reliability": {page.source_id: 1.0 for page in pages},
            "triggered": triggered,
            "extra_passages": extra_passages,
            "margin": margin,
            "coverage_gap": None,
            "mode": "mark",
            "structural": "",
            "decision": "marked" if cross else "plain",
            "nli_pairs": len(table),
            "tags": tags,
        }
    if mode in {"indep", "indep_always", "indep_never", "graph", "bridge"}:
        return _independent_protocol(
            question,
            archive,
            agent_passages,
            local_answers,
            snippets,
            claims,
            table,
            relations,
            mode,
            allow_retrieval,
            allow_conflict,
        )
    reliability = {p.source_id: recency_prior(p.year) for p in union}

    def claim_scores(current_snippets: list[Snippet]) -> list[float]:
        scores = []
        for claim in claims:
            total = 0.0
            for snippet in current_snippets:
                contra, entail, _neutral = table.get((snippet.text, claim), (0.0, 0.0, 1.0))
                if not use_contradiction:
                    contra = 0.0
                weight = reliability[snippet.passage.source_id] if use_reliability else 1.0
                total += weight * snippet.rho * (entail - CONTRA_LAMBDA * contra)
            scores.append(total)
        return scores

    for _ in range(2):
        scores = claim_scores(snippets)
        hat_index = max(range(len(claims)), key=lambda i: scores[i])
        hat = claims[hat_index]
        if not use_reliability:
            break
        grouped: dict[str, list[float]] = {}
        for snippet in snippets:
            contra, entail, _neutral = table.get((snippet.text, hat), (0.0, 0.0, 1.0))
            grouped.setdefault(snippet.passage.source_id, []).append(entail - contra)
        for source_id, values in grouped.items():
            agreement = sum(values) / len(values)
            updated = 1.0 / (1.0 + math.exp(-agreement))
            previous = reliability.get(source_id, 0.5)
            reliability[source_id] = (1.0 - RELIABILITY_ALPHA) * previous + RELIABILITY_ALPHA * updated

    def sentence_role(snippet: Snippet, hat: str) -> tuple[str, float]:
        contra, entail, _neutral = table.get((snippet.text, hat), (0.0, 0.0, 1.0))
        best_other = 0.0
        rival_hit = False
        for claim in claims:
            if claim == hat:
                continue
            _other_contra, other_entail, _other_neutral = table.get(
                (snippet.text, claim), (0.0, 0.0, 1.0)
            )
            best_other = max(best_other, other_entail)
            if other_entail >= RIVAL_ENT and _is_rival_option(question, hat, claim):
                rival_hit = True
        bridge_terms = _question_terms(question, hat)
        mentioned = set(tokenize(snippet.text))
        bridge_hits = sum(1 for term in bridge_terms if term in mentioned)
        if rival_hit and entail < SUPPORT_ENT:
            return "conflict", 0.0
        if entail >= SUPPORT_ENT:
            return "support", entail
        if best_other >= RIVAL_ENT and not rival_hit:
            return "complement", best_other
        if bridge_hits > 0 and contra < 0.75:
            return "bridge", 0.35 + 0.08 * min(bridge_hits, 3)
        if contra > 0.55 and entail < 0.25 and best_other < RIVAL_ENT:
            return "conflict", 0.0
        return "residual", max(entail, 0.15)

    def choose_bundle(current_snippets: list[Snippet], hat: str) -> list[Snippet]:
        def flat_ranked() -> list[Snippet]:
            ranked = []
            for snippet in current_snippets:
                contra, entail, _neutral = table.get((snippet.text, hat), (0.0, 0.0, 1.0))
                if use_contradiction and contra > 0.55 and entail < 0.25:
                    continue
                weight = reliability[snippet.passage.source_id] if use_reliability else 1.0
                ranked.append((weight * snippet.rho * max(entail, 0.15), snippet))
            ranked.sort(key=lambda item: item[0], reverse=True)
            return [snippet for _score, snippet in ranked[:BUNDLE_K]] or current_snippets[:BUNDLE_K]

        if mode == "cover":
            # One strong sentence per live claim, then fill from the leading claim.
            # Rival answers stay visible instead of being deleted as contradictions.
            chosen: list[Snippet] = []
            seen_text: set[str] = set()
            for claim in claims:
                best_snippet = None
                best_score = 0.35
                for snippet in current_snippets:
                    if snippet.text in seen_text:
                        continue
                    contra, entail, _neutral = table.get((snippet.text, claim), (0.0, 0.0, 1.0))
                    if use_contradiction and contra > 0.70 and entail < 0.20:
                        continue
                    weight = reliability[snippet.passage.source_id] if use_reliability else 1.0
                    score = weight * snippet.rho * entail
                    if score > best_score:
                        best_score = score
                        best_snippet = snippet
                if best_snippet is not None:
                    seen_text.add(best_snippet.text)
                    chosen.append(best_snippet)
            for snippet in flat_ranked():
                if len(chosen) >= BUNDLE_K:
                    break
                if snippet.text in seen_text:
                    continue
                seen_text.add(snippet.text)
                chosen.append(snippet)
            return chosen[:BUNDLE_K] or current_snippets[:BUNDLE_K]

        if mode not in ("typed", "drop"):
            return flat_ranked()

        grouped: dict[str, list[tuple[float, Snippet]]] = {
            "support": [],
            "complement": [],
            "bridge": [],
            "residual": [],
            "conflict": [],
        }
        for snippet in current_snippets:
            role, strength = sentence_role(snippet, hat)
            if role == "conflict" and use_contradiction:
                grouped["conflict"].append((0.0, snippet))
                continue
            weight = reliability[snippet.passage.source_id] if use_reliability else 1.0
            grouped[role].append((weight * snippet.rho * strength, snippet))
        for role in grouped:
            grouped[role].sort(key=lambda item: item[0], reverse=True)
        chosen: list[Snippet] = []
        seen_text = set()

        def take(pool: list[tuple[float, Snippet]]) -> None:
            for _score, snippet in pool:
                if len(chosen) >= BUNDLE_K or snippet.text in seen_text:
                    continue
                seen_text.add(snippet.text)
                chosen.append(snippet)

        take(grouped["support"])
        take(grouped["complement"])
        take(grouped["bridge"])
        take(grouped["residual"])
        if not chosen:
            take([(0.0, snippet) for snippet in current_snippets[:BUNDLE_K]])
        return chosen[:BUNDLE_K]

    scores = claim_scores(snippets)
    beliefs = softmax(scores)
    order = sorted(range(len(claims)), key=lambda i: beliefs[i], reverse=True)
    hat = claims[order[0]]
    margin = 1.0 if len(order) == 1 else beliefs[order[0]] - beliefs[order[1]]
    bundle = choose_bundle(snippets, hat)
    support_mass = 0.0
    if bundle:
        masses = []
        for snippet in bundle:
            _contra, entail, _neutral = table.get((snippet.text, hat), (0.0, 0.0, 1.0))
            weight = reliability[snippet.passage.source_id] if use_reliability else 1.0
            masses.append(weight * entail)
        support_mass = sum(masses) / len(masses)
    if mode == "typed":
        terms = _question_terms(question, "")
        if terms:
            covered = 0
            bundle_tokens = [set(tokenize(snippet.text)) for snippet in bundle]
            for term in terms:
                if any(term in tokens for tokens in bundle_tokens):
                    covered += 1
            coverage_gap = 1.0 - covered / len(terms)
        else:
            coverage_gap = 0.0
        unresolved = False
        if len(order) > 1:
            rival = claims[order[1]]
            held_text = {snippet.text for snippet in bundle}
            for snippet in snippets:
                if snippet.text in held_text:
                    continue
                _rival_contra, rival_entail, _rival_neutral = table.get(
                    (snippet.text, rival), (0.0, 0.0, 1.0)
                )
                _hat_contra, hat_entail, _hat_neutral = table.get(
                    (snippet.text, hat), (0.0, 0.0, 1.0)
                )
                if rival_entail >= RIVAL_ENT and hat_entail < SUPPORT_ENT:
                    unresolved = True
                    break
        triggered = allow_retrieval and (
            support_mass < SUPPORT_FLOOR
            or coverage_gap >= COVER_GAP
            or (margin < MARGIN_THRESHOLD and unresolved)
        )
    else:
        coverage_gap = None
        triggered = allow_retrieval and (margin < MARGIN_THRESHOLD or support_mass < SUPPORT_FLOOR)
    extra_passages = []
    if triggered:
        held = {snippet.passage.pid for snippet in bundle}
        pool = [p for p in archive if p.pid.startswith("p") and p.pid not in held]
        if pool:
            query = question + " " + hat
            if len(order) > 1:
                query += " " + claims[order[1]]
            corpus = [tokenize(p.title + " " + p.text) or ["empty"] for p in pool]
            ranker = BM25Okapi(corpus)
            raw_scores = ranker.get_scores(tokenize(query) or ["empty"])
            best = pool[max(range(len(pool)), key=lambda i: float(raw_scores[i]))]
            extra_passages = [best]
            extra_snips = _rank_sentences(question, [best])[:2]
            if extra_snips:
                extra_table = relations(extra_snips, claims)
                table.update(extra_table)
                for snippet in extra_snips:
                    reliability.setdefault(snippet.passage.source_id, recency_prior(snippet.passage.year))
                snippets = snippets + extra_snips
                scores = claim_scores(snippets)
                beliefs = softmax(scores)
                order = sorted(range(len(claims)), key=lambda i: beliefs[i], reverse=True)
                hat = claims[order[0]]
                bundle = choose_bundle(snippets, hat)

    return {
        "claims": claims,
        "beliefs": beliefs,
        "hat": hat,
        "bundle": bundle,
        "reliability": reliability,
        "triggered": triggered,
        "extra_passages": extra_passages,
        "margin": margin,
        "coverage_gap": coverage_gap,
        "mode": reported_mode if reported_mode == "gate" else mode,
        "structural": "",
        "decision": decision_label,
        "nli_pairs": len(table),
    }


def _independent_protocol(
    question,
    archive,
    agent_passages,
    local_answers,
    snippets,
    claims,
    table,
    relations,
    mode,
    allow_retrieval,
    allow_conflict,
):
    """A claim counts only when another view states it.

    Reliability is not moved toward the leading claim. A single externally
    corroborated claim is the answer, so the reader does not decide it.
    Two self-supported claims with no external corroboration are a conflict
    when the task allows that label. Retrieval runs only when nothing is
    corroborated, and the query is the question rather than the leader.
    """
    triggered = False
    extra_passages = []
    if mode == "indep_always":
        held = {snippet.passage.pid for snippet in snippets}
        found = _question_lookup(question, archive, held)
        if found is not None:
            extra_passages = [found]
            extra_snips = _rank_sentences(question, [found])[:2]
            if extra_snips:
                table.update(relations(extra_snips, claims))
                snippets = snippets + extra_snips
            triggered = True

    def pack(found, self_entail, current_snippets):
        corroborated = [(found[claim][0], claim, found[claim][1]) for claim in claims if found[claim]]
        corroborated.sort(key=lambda item: item[0], reverse=True)
        self_supported = [claim for claim in claims if self_entail[claim] >= SUPPORT_ENT]
        external_scores = [found[claim][0] if found[claim] else 0.0 for claim in claims]
        if len(corroborated) == 1:
            _score, claim, witness = corroborated[0]
            bundle = [witness]
            for snippet in current_snippets:
                if len(bundle) >= BUNDLE_K or snippet.text == witness.text:
                    continue
                if _stated_in(snippet.text, claim):
                    bundle.append(snippet)
            return claim, "corroborated", bundle[:BUNDLE_K], external_scores
        if len(corroborated) >= 2 and allow_conflict:
            bundle = [item[2] for item in corroborated[:BUNDLE_K]]
            return "conflict", "conflict", bundle, external_scores
        if len(corroborated) >= 2:
            bundle = [item[2] for item in corroborated[:BUNDLE_K]]
            return "", "reader", bundle, external_scores
        if allow_conflict and len(self_supported) >= 2:
            return "conflict", "conflict", current_snippets[:BUNDLE_K], external_scores
        return "", "seek", current_snippets[:BUNDLE_K], external_scores

    if mode == "graph":
        owners = _proposer_sources(local_answers, agent_passages, claims)
        found, self_entail = _external_hits(snippets, claims, owners, table)
        external_scores = [found[claim][0] if found[claim] else 0.0 for claim in claims]
        bundle = _graph_bundle(snippets, claims, table)
        structural, decision = "", "reader"
    else:
        owners = _proposer_sources(local_answers, agent_passages, claims)
        found, self_entail = _external_hits(snippets, claims, owners, table)

        def consider_bridge(current_snippets):
            nonlocal structural, decision, bundle
            if mode != "bridge" or decision != "seek":
                return
            bridged = _bridge_hits(current_snippets, claims, owners, table)
            hits = [(claim, bridged[claim]) for claim in claims if bridged[claim]]
            hits.sort(key=lambda item: item[1][0], reverse=True)
            if len(hits) == 1:
                claim, (_score, witness) = hits[0]
                structural = claim
                decision = "bridge"
                bundle = [witness]
            elif len(hits) >= 2 and allow_conflict:
                structural = "conflict"
                decision = "conflict"
                bundle = [item[1][1] for item in hits[:BUNDLE_K]]
            elif len(hits) >= 2:
                decision = "reader"
                bundle = [item[1][1] for item in hits[:BUNDLE_K]]

        structural, decision, bundle, external_scores = pack(found, self_entail, snippets)
        consider_bridge(snippets)
        if decision == "seek" and allow_retrieval and mode in {"indep", "bridge"}:
            held = {snippet.passage.pid for snippet in snippets}
            found_page = _question_lookup(question, archive, held)
            triggered = True
            if found_page is not None:
                extra_passages = [found_page]
                extra_snips = _rank_sentences(question, [found_page])[:2]
                if extra_snips:
                    table.update(relations(extra_snips, claims))
                    snippets = snippets + extra_snips
                    found, self_entail = _external_hits(snippets, claims, owners, table)
                    structural, decision, bundle, external_scores = pack(found, self_entail, snippets)
                    consider_bridge(snippets)
            if decision == "seek":
                decision = "reader"
        elif decision == "seek":
            decision = "reader"
        if decision == "reader":
            kept = []
            for snippet in snippets:
                pure_self = False
                external_ok = False
                for claim in claims:
                    _contra, entail, _neutral = table.get((snippet.text, claim), (0.0, 0.0, 1.0))
                    if entail < SUPPORT_ENT or not _stated_in(snippet.text, claim):
                        continue
                    if snippet.passage.source_id in owners.get(claim, set()):
                        pure_self = True
                    else:
                        external_ok = True
                if external_ok or not pure_self:
                    kept.append(snippet)
            bundle = (kept or snippets)[:BUNDLE_K]

    beliefs = softmax(external_scores)
    order = sorted(range(len(claims)), key=lambda i: beliefs[i], reverse=True)
    hat = claims[order[0]]
    margin = 1.0 if len(order) == 1 else beliefs[order[0]] - beliefs[order[1]]
    reliability = {}
    for snippet in snippets:
        reliability[snippet.passage.source_id] = 1.0
    return {
        "claims": claims,
        "beliefs": beliefs,
        "hat": hat,
        "bundle": bundle,
        "reliability": reliability,
        "triggered": triggered,
        "extra_passages": extra_passages,
        "margin": margin,
        "coverage_gap": None,
        "mode": mode,
        "structural": structural,
        "decision": decision,
        "nli_pairs": len(table),
    }


def claim_hypothesis(question: str, claim: str) -> str:
    # A short question-answer hypothesis. "The answer is X" is mostly scored as
    # neutral by the MNLI model even when the evidence states X.
    return f"{question.strip()} {claim.strip()}"


POLAR_PREFIXES = (
    "are ",
    "is ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "has ",
    "have ",
)


def is_polar_question(question: str) -> bool:
    text = (question or "").strip().lower()
    return text.startswith(POLAR_PREFIXES)


def canonicalize_polar(question: str, answer: str) -> str:
    """Bind a polar reader span to the yes/no lexicon the questions use."""
    if not is_polar_question(question):
        return (answer or "").strip()
    key = re.sub(r"\s+", " ", (answer or "").strip().lower())
    key = key.strip(" .\"'")
    if key in {"true", "yes"}:
        return "yes"
    if key in {"false", "no"}:
        return "no"
    return (answer or "").strip()


def joint_snippets(agent_passages: list[list[Passage]]) -> list[str]:
    """Sentences from every local page, in agent order, without duplicates."""
    snippets: list[str] = []
    seen: set[str] = set()
    for group in agent_passages:
        for passage in group:
            for sentence in split_sentences(passage.text):
                if sentence in seen:
                    continue
                seen.add(sentence)
                snippets.append(sentence)
    return snippets


def cross_support(nli: NLI, question: str, snippets: list[str], answer: str) -> tuple[int, float]:
    """Rank key of one candidate on the joint evidence.

    A sentence that states the answer outranks every sentence that does not.
    Inside that class the strongest entailment-minus-contradiction wins.
    ``stated_in`` keeps the existing 0.5 token-recall test. No new cutoff.
    """
    text = (answer or "").strip()
    if not snippets or not text or text.lower() == "conflict":
        return (0, -1.0)
    pairs = [(snippet, claim_hypothesis(question, text)) for snippet in snippets]
    scored = nli.score(pairs)
    best_stated = None
    best_any = -1.0
    for snippet, (contra, entail, _neutral) in zip(snippets, scored):
        margin = float(entail - contra)
        if margin > best_any:
            best_any = margin
        if _stated_in(snippet, text):
            if best_stated is None or margin > best_stated:
                best_stated = margin
    if best_stated is not None:
        return (1, float(best_stated))
    return (0, float(best_any))


def select_cross(nli: NLI, question: str, snippets: list[str], pool_answer: str, witness_answer: str) -> tuple[str, str]:
    """Keep the pool span unless the witness span has strictly higher joint support.

    Equivalent spans stay with the pool. The comparison is the cross-check:
    the full page set and the contradiction-filtered witness are both read,
    and the joint sentences adjudicate the disagreement.
    """
    pool = canonicalize_polar(question, pool_answer)
    witness = canonicalize_polar(question, witness_answer)
    if token_f1(pool, witness) >= 0.999 or pool == witness:
        return pool, "agree"
    if cross_support(nli, question, snippets, witness) > cross_support(nli, question, snippets, pool):
        return witness, "witness"
    return pool, "pool"


def bundle_support(nli: NLI, question: str, snippets: list[str], answer: str) -> float:
    """Mean entailment minus contradiction of one answer on the shared bundle."""
    if not snippets or not (answer or "").strip():
        return 0.0
    pairs = [(text, claim_hypothesis(question, answer)) for text in snippets]
    scored = nli.score(pairs)
    if not scored:
        return 0.0
    return float(sum(entail - contra for contra, entail, _neutral in scored) / len(scored))


def commit_answer(
    nli: NLI,
    question: str,
    evidence: list[str],
    local_answers: list[str],
    reader_answer: str,
) -> str:
    """Let the reader overturn a unanimous committee only with stronger support.

    Alternatives stay in the bundle, so the reader can still correct a wrong
    committee when a sentence actually favors the new span. An unsupported flip
    back to the rival option is rejected.
    """
    reader = canonicalize_polar(question, reader_answer)
    canon_locals = [canonicalize_polar(question, answer) for answer in local_answers]
    keys = []
    for answer in canon_locals:
        key = re.sub(r"\s+", " ", answer.strip().lower())
        if key:
            keys.append(key)
    if len(keys) < 2 or len(set(keys)) != 1:
        return reader
    agreed = canon_locals[0]
    if token_f1(reader, agreed) >= 0.5:
        return reader
    if bundle_support(nli, question, evidence, reader) > bundle_support(
        nli, question, evidence, agreed
    ):
        return reader
    return agreed


def support_confidence(
    nli: NLI,
    question: str,
    snippets: list[str],
    answer: str,
    weights: list[float] | None = None,
) -> float:
    if not snippets or not answer:
        return 0.0
    pairs = [(text, claim_hypothesis(question, answer)) for text in snippets]
    scored = nli.score(pairs)
    if weights is None:
        weights = [1.0] * len(snippets)
    num = 0.0
    den = 0.0
    for (_contra, entail, _neutral), weight in zip(scored, weights):
        num += max(weight, 0.0) * entail
        den += max(weight, 0.0)
    if den <= 0:
        return 0.0
    return float(max(0.0, min(1.0, num / den)))
