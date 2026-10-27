"""A derivation is the answer. Each step must be stated in the evidence.

The plan is written from the question alone, before any page is read, so a
corrupted page cannot choose the lookups. A chain answer is the last stated
span. A comparison proves one value for each entity, and that value may itself
be one bridge step. The comparison is then decided from those two values.
If a step is not stated, the derivation is empty and the witness reader answers.

This procedure was frozen before the folio block was scored.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from epirag.data import Passage, tokenize
from epirag.execute import decide, direction_later, parse_year, split_pair

_REJECT = {"", "unknown", "none", "n/a", "unanswerable"}
_LABEL = re.compile(r"^(Q1B|Q2B|E1|E2|Q1|Q2)\s*[:\-\.]\s*(.+)$", re.IGNORECASE)
_LABEL_FIND = re.compile(
    r"(?is)\b(Q1B|Q2B|E1|E2|Q1|Q2)\s*[:.\-]\s*(.*?)(?=\b(?:Q1B|Q2B|E1|E2|Q1|Q2)\s*[:.\-]|$)"
)
_BAD_LOOKUP = re.compile(r"\b(first|older|younger|earlier|later|same)\b", re.IGNORECASE)
_WH = re.compile(r"^(who|what|when|where|which|in|the|a|an)$", re.IGNORECASE)
_LOOKUP_NOTE = "Copy the answer span from the evidence. Do not write a passage number."
_PLAN = """Output only lookup lines. Do not explain. PERSON means the previous answer.

Question: Which university was founded first, Duke or Tufts?
COMPARE
E1: Duke
Q1: In what year was Duke founded?
E2: Tufts
Q2: In what year was Tufts founded?

Question: Where did the director of Film die?
CHAIN
Q1: Who directed Film?
Q2: Where did PERSON die?

Question: Which film has the director who died first, Film A or Film B?
COMPARE
E1: Film A
Q1: Who directed Film A?
Q1B: In what year did PERSON die?
E2: Film B
Q2: Who directed Film B?
Q2B: In what year did PERSON die?

Question: {question}
"""


def plan_prompt(question: str) -> str:
    return _PLAN.format(question=" ".join((question or "").split()))


def _clean_line(raw: str) -> str:
    line = re.sub(r"\*+", "", raw.strip())
    line = re.sub(r"^\d+\s*[\.\)\]]\s*", "", line)
    return " ".join(line.split())


def _lookup_ok(question: str) -> bool:
    text = " ".join((question or "").split())
    if len(text.split()) < 2 or _BAD_LOOKUP.search(text):
        return False
    return True


def _field_value(raw: str) -> str:
    text = " ".join((raw or "").replace("\n", " ").split())
    text = text.strip(" \"'`")
    text = re.sub(r"^(question|lookup)\s*\d*\s*[:.\-]\s*", "", text, flags=re.IGNORECASE)
    if "?" in text:
        text = text[: text.find("?") + 1]
    return text[:160].strip()


def _fields_of(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    lookups: list[str] = []
    for match in _LABEL_FIND.finditer(text or ""):
        key = match.group(1).upper()
        value = _field_value(match.group(2))
        if not value:
            continue
        if key == "Q1":
            lookups.append(value)
            continue
        if key not in fields:
            fields[key] = value
    if lookups:
        fields["Q1"] = lookups[0]
    if len(lookups) > 1 and "Q2" not in fields:
        fields["Q2"] = lookups[1]
    return fields


def _expand_entity(name: str, question: str) -> str:
    """Keep the entity span the question itself uses."""
    needle = " ".join((name or "").split())
    if len(needle) < 2 or not question:
        return needle
    match = re.search(re.escape(needle) + r"(?:\s+[A-Z][\w'’.-]*)*", question)
    if match and len(match.group(0)) > len(needle):
        return match.group(0)
    return needle


def _covers(lookup: str, question: str) -> bool:
    """A one-step proof must still ask the original question."""
    from epirag.metrics import normalize_answer

    asked = [token for token in normalize_answer(question).split() if len(token) > 4]
    have = set(normalize_answer(lookup).split())
    if not asked:
        return True
    return sum(token in have for token in asked) / len(asked) >= 0.5


def _lookups_ok(questions: tuple[str, ...]) -> bool:
    for question in questions:
        if question and not _lookup_ok(question):
            return False
    return True


def parse_plan(text: str, question: str = "") -> dict | None:
    fields = _fields_of(text)
    question1 = fields.get("Q1", "")
    question2 = fields.get("Q2", "")
    question1b = fields.get("Q1B", "")
    question2b = fields.get("Q2B", "")
    if not _lookups_ok((question1, question2, question1b, question2b)):
        return None
    if not question1:
        return None
    entity1 = fields.get("E1", "")
    entity2 = fields.get("E2", "")
    person_step = bool(re.search(r"\bPERSON\b", question2, flags=re.IGNORECASE))
    kind_word = None
    for line in (text or "").splitlines():
        key = _clean_line(line).lower().rstrip(":")
        if key in {"compare", "chain"}:
            kind_word = key
            break
    if kind_word == "chain" or (person_step and kind_word != "compare"):
        if not question2 and question and not _covers(question1, question):
            return None
        return {"kind": "chain", "question1": question1, "question2": question2}
    if question2 and (kind_word == "compare" or (entity1 and entity2) or person_step):
        pair = (entity1, entity2) if entity1 and entity2 else split_pair(question)
        if pair is None:
            return {"kind": "chain", "question1": question1, "question2": question2}
        entity1, entity2 = pair
        entity1 = _expand_entity(entity1, question)
        entity2 = _expand_entity(entity2, question)
        if min(len(entity1), len(entity2)) < 2 or entity1.lower() == entity2.lower():
            return None
        return {
            "kind": "compare",
            "entity1": entity1[:80],
            "question1": question1,
            "question1b": question1b,
            "entity2": entity2[:80],
            "question2": question2,
            "question2b": question2b,
        }
    if question2 and split_pair(question):
        entity1, entity2 = split_pair(question)
        return {
            "kind": "compare",
            "entity1": entity1[:80],
            "question1": question1,
            "question1b": question1b,
            "entity2": entity2[:80],
            "question2": question2,
            "question2b": question2b,
        }
    if not question2:
        if question and not _covers(question1, question):
            return None
        return {"kind": "chain", "question1": question1, "question2": ""}
    return {"kind": "chain", "question1": question1, "question2": question2}


def _acceptable(answer: str) -> bool:
    text = " ".join((answer or "").split())
    key = text.lower().strip(" .")
    if key in _REJECT or key.startswith("no information"):
        return False
    if re.fullmatch(r"\[\d+\]", key):
        return False
    if len(text.split()) > 12:
        return False
    return True


def _snippet(page: Passage, sentence: str) -> Passage:
    return Passage(
        pid=page.pid,
        title=page.title,
        text=sentence,
        gold=page.gold,
        year=page.year,
        source_id=page.source_id,
    )


def _about(entity: str, pages: list[Passage], limit: int = 4) -> list[Passage]:
    """Sentences that mention the entity this step is about."""
    from epirag.engine import split_sentences

    needle = " ".join((entity or "").lower().split())
    if len(needle) < 2:
        return []
    mentioned: list[Passage] = []
    titled: list[Passage] = []
    seen: set[str] = set()
    for page in pages:
        title_hit = needle in page.title.lower()
        for sentence in split_sentences(page.text):
            if len(sentence.split()) < 3 or sentence in seen:
                continue
            snippet = _snippet(page, sentence)
            if needle in sentence.lower():
                seen.add(sentence)
                mentioned.append(snippet)
            elif title_hit:
                seen.add(sentence)
                titled.append(snippet)
    chosen = mentioned[:limit]
    if len(chosen) < limit:
        chosen.extend(titled[: limit - len(chosen)])
    return chosen


def _focus(query: str, pages: list[Passage], limit: int = 4) -> list[Passage]:
    from epirag.engine import split_sentences

    sentences: list[Passage] = []
    for page in pages:
        for sentence in split_sentences(page.text):
            if len(sentence.split()) < 3:
                continue
            sentences.append(_snippet(page, sentence))
    if not sentences:
        return []
    corpus = [tokenize(item.title + " " + item.text) or ["empty"] for item in sentences]
    ranker = BM25Okapi(corpus)
    scores = ranker.get_scores(tokenize(query) or ["empty"])
    order = sorted(range(len(sentences)), key=lambda i: float(scores[i]), reverse=True)
    chosen: list[Passage] = []
    seen: set[str] = set()
    for index in order:
        text = sentences[index].text
        if text in seen:
            continue
        seen.add(text)
        chosen.append(sentences[index])
        if len(chosen) >= limit:
            break
    return chosen


def _proper_name(text: str) -> str:
    spans = re.findall(r"\b[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*)+", text or "")
    spans = [span for span in spans if not _WH.match(span.split()[0])]
    if not spans:
        return ""
    return max(spans, key=len)


def _evidence(query: str, pages: list[Passage], entity: str) -> list[Passage]:
    needle = " ".join((entity or "").split()) or _proper_name(query)
    if needle:
        found = _about(needle, pages)
        if found:
            return found
    return _focus(query, pages)


def _stated(answer: str, pages: list[Passage]) -> bool:
    from epirag.metrics import normalize_answer

    gold = normalize_answer(answer).split()
    if not gold:
        return False
    have: set[str] = set()
    for page in pages:
        have.update(normalize_answer(page.text).split())
    return sum(token in have for token in gold) / len(gold) >= 0.5


def _same_span(left: str, right: str) -> bool:
    from epirag.metrics import normalize_answer, token_f1

    if not left or not right:
        return False
    if normalize_answer(left) == normalize_answer(right):
        return True
    return token_f1(left, right) >= 0.5


def _sealed_prompt(question: str, entity1: str, value1: str, entity2: str, value2: str) -> str:
    return (
        "Answer using only these two facts. "
        "Reply with one of the two names. "
        "Do not explain.\n"
        f"Fact 1: {entity1}: {value1}\n"
        f"Fact 2: {entity2}: {value2}\n"
        f"Question: {question}"
    )


def _generate(generate, prompts: list[str], max_new_tokens: int) -> list[str]:
    if not prompts:
        return []
    try:
        return generate(prompts, max_new_tokens=max_new_tokens)
    except TypeError:
        return generate(prompts)


def resolve_derivations(generate, items: list[tuple[str, list]], count=None) -> list[dict]:
    """Return a proved span, or an empty answer when the witness reader should answer."""
    from epirag.engine import evidence_prompt, parse_answer

    def blank() -> dict:
        return {"answer": "", "decision": "witness", "hop_tokens": 0}

    def add_tokens(index: int, prompt: str) -> None:
        if count is not None and prompt:
            tokens[index] = tokens.get(index, 0) + count(prompt)

    def proved_span(raw: str, evidence: list[Passage]) -> str:
        answer, _quote = parse_answer(raw)
        answer = " ".join(answer.split())
        if _acceptable(answer) and _stated(answer, evidence):
            return answer
        return ""

    results = [blank() for _item in items]
    tokens: dict[int, int] = {}
    plans = []
    plan_prompts = [plan_prompt(question) for question, _pages in items]
    for index, prompt in enumerate(plan_prompts):
        add_tokens(index, prompt)
    raw_plans = _generate(generate, plan_prompts, 128)
    for index, ((_question, pages), raw) in enumerate(zip(items, raw_plans)):
        spec = parse_plan(raw, _question)
        if spec is not None:
            plans.append((index, pages, spec))

    first_prompts: list[str] = []
    first_meta: list[tuple] = []
    for index, pages, spec in plans:
        if spec["kind"] == "chain":
            evidence = _evidence(spec["question1"], pages, "")
            prompt = evidence_prompt(spec["question1"], evidence, _LOOKUP_NOTE) if evidence else ""
            first_prompts.append(prompt)
            first_meta.append((index, "chain1", prompt, evidence, spec))
            continue
        for slot, entity, question in (
            ("left", spec["entity1"], spec["question1"]),
            ("right", spec["entity2"], spec["question2"]),
        ):
            evidence = _evidence(question, pages, entity)
            prompt = evidence_prompt(question, evidence, _LOOKUP_NOTE) if evidence else ""
            first_prompts.append(prompt)
            first_meta.append((index, slot, prompt, evidence, spec))

    raw_first = _generate(generate, [prompt for prompt in first_prompts if prompt], 64)
    cursor = 0
    values: dict[tuple[int, str], str] = {}
    second_prompts: list[str] = []
    second_meta: list[tuple] = []
    for index, slot, prompt, evidence, spec in first_meta:
        if not prompt:
            continue
        add_tokens(index, prompt)
        answer = proved_span(raw_first[cursor], evidence)
        cursor += 1
        if not answer:
            continue
        if slot == "chain1":
            if not spec["question2"]:
                results[index] = {
                    "answer": answer,
                    "decision": "chain",
                    "hop_tokens": tokens.get(index, 0),
                }
                continue
            asked = re.sub(r"\bPERSON\b", answer, spec["question2"], flags=re.IGNORECASE)
            pages = next(pages for item_index, pages, _spec in plans if item_index == index)
            evidence2 = _evidence(asked, pages, answer)
            prompt2 = evidence_prompt(asked, evidence2, _LOOKUP_NOTE) if evidence2 else ""
            second_prompts.append(prompt2)
            second_meta.append((index, "chain2", prompt2, evidence2, spec))
            continue
        further = spec["question1b"] if slot == "left" else spec["question2b"]
        if not further:
            values[(index, slot)] = answer
            continue
        asked = re.sub(r"\bPERSON\b", answer, further, flags=re.IGNORECASE)
        pages = next(pages for item_index, pages, _spec in plans if item_index == index)
        evidence2 = _evidence(asked, pages, answer)
        prompt2 = evidence_prompt(asked, evidence2, _LOOKUP_NOTE) if evidence2 else ""
        second_prompts.append(prompt2)
        second_meta.append((index, slot, prompt2, evidence2, spec))

    raw_second = _generate(generate, [prompt for prompt in second_prompts if prompt], 64)
    cursor = 0
    for index, slot, prompt, evidence, _spec in second_meta:
        if not prompt:
            continue
        add_tokens(index, prompt)
        answer = proved_span(raw_second[cursor], evidence)
        cursor += 1
        if not answer:
            continue
        if slot == "chain2":
            results[index] = {
                "answer": answer,
                "decision": "chain",
                "hop_tokens": tokens.get(index, 0),
            }
        else:
            values[(index, slot)] = answer

    for index, _pages, spec in plans:
        if spec["kind"] != "compare":
            continue
        left = values.get((index, "left"), "")
        right = values.get((index, "right"), "")
        if not left or not right:
            continue
        question = items[index][0]
        if re.search(r"\bsame\b", question, flags=re.IGNORECASE):
            results[index] = {
                "answer": "yes" if _same_span(left, right) else "no",
                "decision": "compare",
                "hop_tokens": tokens.get(index, 0),
            }
            continue
        later = direction_later(question)
        left_year = parse_year(left)
        right_year = parse_year(right)
        if later is not None and left_year is not None and right_year is not None:
            chosen = decide(spec["entity1"], spec["entity2"], left_year, right_year, later)
            if chosen:
                results[index] = {
                    "answer": chosen,
                    "decision": "compare",
                    "hop_tokens": tokens.get(index, 0),
                }
            continue
        if _same_span(left, right):
            continue
        prompt = _sealed_prompt(question, spec["entity1"], left, spec["entity2"], right)
        second_prompts.append(prompt)
        second_meta.append((index, "sealed", prompt, [], spec))

    # Sealed prompts were appended after the second generation. Generate only those.
    sealed_prompts = [prompt for index, slot, prompt, _shown, _spec in second_meta if slot == "sealed" and prompt]
    raw_sealed = _generate(generate, sealed_prompts, 64)
    cursor = 0
    for index, slot, prompt, _shown, spec in second_meta:
        if slot != "sealed" or not prompt:
            continue
        add_tokens(index, prompt)
        answer, _quote = parse_answer(raw_sealed[cursor])
        cursor += 1
        answer = " ".join(answer.split())
        if _same_span(answer, spec["entity1"]) and not _same_span(answer, spec["entity2"]):
            kept = spec["entity1"]
        elif _same_span(answer, spec["entity2"]) and not _same_span(answer, spec["entity1"]):
            kept = spec["entity2"]
        else:
            continue
        results[index] = {
            "answer": kept,
            "decision": "compare",
            "hop_tokens": tokens.get(index, 0),
        }
    for index, _pages, _spec in plans:
        results[index]["hop_tokens"] = tokens.get(index, results[index]["hop_tokens"])
    return results


def _self_check() -> None:
    chain = parse_plan("CHAIN\nQ1: Who directed Film?\nQ2: Where did PERSON die?")
    assert chain is not None and chain["question2"].startswith("Where"), chain
    compare = parse_plan(
        "Note\nCOMPARE\nE1: Ann\nQ1: When was Ann born?\nE2: Bea\nQ2: When was Bea born?"
    )
    assert compare is not None and compare["entity2"] == "Bea", compare
    nested = parse_plan(
        "COMPARE\nE1: Film A\nQ1: Who directed Film A?\nQ1B: In what year did PERSON die?\n"
        "E2: Film B\nQ2: Who directed Film B?\nQ2B: In what year did PERSON die?"
    )
    assert nested is not None and nested["question1b"].startswith("In what year"), nested
    assert parse_plan("COMPARE\nE1: Duke\nQ1: Which was founded first?\nE2: Tufts\nQ2: In what year was Tufts founded?") is None
    assert decide("Ann", "Bea", 1980, 1990, False) == "Ann"
    assert parse_plan("hello") is None
    inline = parse_plan(
        "Q1: In what year was Ann born? Q2: In what year was Bea born?",
        "Who is older, Ann or Bea?",
    )
    assert inline is not None and inline["kind"] == "compare" and inline["entity1"] == "Ann", inline
    buried = parse_plan(
        'by asking: Q1: Who directed Film? Q2: Where did PERSON die?',
        "Where did the director of Film die?",
    )
    assert buried is not None and buried["kind"] == "chain" and "PERSON" in buried["question2"], buried
    duplicate = parse_plan(
        "E1: Duke University\nQ1: In what year was Duke University founded?\n"
        "E2: Tufts University\nQ1: In what year was Tufts University founded?",
        "which university was founded first Duke University or Tufts University?",
    )
    assert duplicate is not None and duplicate["kind"] == "compare" and "Tufts" in duplicate["question2"], duplicate
    short = parse_plan(
        "E1: Duke\nQ1: In what year was Duke founded?\nE2: Tufts\nQ2: In what year was Tufts founded?",
        "which university was founded first Duke University or Tufts University?",
    )
    assert short is not None and short["entity1"] == "Duke University" and short["entity2"] == "Tufts University", short
    assert parse_plan(
        "Q1: Who performed the song Lovers Again?",
        "Which country the performer of song Lovers Again is from?",
    ) is None

    pages = [
        Passage("p1", "Film", "John Smith directed the film in Paris.", True, 2020, "a0"),
        Passage("p2", "Ann", "Ann was born in 1980.", True, 2020, "a1"),
        Passage("p3", "Bea", "Bea was born in 1990.", True, 2020, "a2"),
    ]

    def stub(prompts, max_new_tokens=64):
        del max_new_tokens
        out = []
        for prompt in prompts:
            if prompt.startswith("Output only lookup lines."):
                if "Question: Who is older" in prompt:
                    out.append(
                        "COMPARE\nE1: Ann\nQ1: When was Ann born?\nE2: Bea\nQ2: When was Bea born?"
                    )
                else:
                    out.append("CHAIN\nQ1: Who directed Film?\nQ2: Where did PERSON die?")
            elif "Who directed Film" in prompt:
                out.append("John Smith")
            elif "Where did John Smith die" in prompt:
                out.append("Paris")
            elif "When was Ann born" in prompt:
                out.append("1980")
            elif "When was Bea born" in prompt:
                out.append("1990")
            else:
                out.append("unknown")
        return out

    chain_out, compare_out = resolve_derivations(
        stub,
        [
            ("Where did the director of Film die?", pages),
            ("Who is older, Ann or Bea?", pages),
        ],
    )
    assert chain_out["decision"] == "chain" and chain_out["answer"] == "Paris", chain_out
    assert compare_out["decision"] == "compare" and compare_out["answer"] == "Ann", compare_out
    print("derive self-check ok")


if __name__ == "__main__":
    _self_check()
