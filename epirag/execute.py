"""Execute a two-entity comparison outside the reader.

The reader answers only a one-hop year question. The coordinator compares the
two years. If either year is missing, the caller keeps the witness answer.
This rule is frozen before any score on a new slice is read.
"""

from __future__ import annotations

import re

from epirag.data import Passage
from epirag.engine import evidence_prompt, parse_answer, split_sentences

YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
_OR = re.compile(r"\s+or\s+", re.IGNORECASE)
_CUE = re.compile(
    r"^.*\b(?:first|earlier|later|before|after|between)\b\s+",
    re.IGNORECASE,
)


def split_pair(question: str) -> tuple[str, str] | None:
    text = question.strip().rstrip("?").strip()
    parts = _OR.split(text, maxsplit=1)
    if len(parts) != 2:
        return None
    right = parts[1].strip(" .\"'")
    left = parts[0]
    if "," in left:
        left = left.rsplit(",", 1)[-1]
    else:
        left = _CUE.sub("", left, count=1)
    left = left.strip(" .\"'")
    if len(left) < 2 or len(right) < 2:
        return None
    if left.lower() == right.lower():
        return None
    return left, right


def direction_later(question: str) -> bool | None:
    text = question.lower()
    later = bool(re.search(r"\b(later|after|younger)\b", text))
    earlier = bool(re.search(r"\b(earlier|first|before|older)\b", text))
    if later == earlier:
        return None
    return later


def attribute(question: str) -> str | None:
    text = question.lower()
    if "director" in text and re.search(r"\bborn\b", text):
        return "director-born"
    if "director" in text and re.search(r"\b(died|dies|die)\b", text):
        return "director-died"
    if re.search(r"\bborn\b", text):
        return "born"
    if re.search(r"\b(died|dies|die)\b", text):
        return "died"
    if "came out" in text or re.search(
        r"\b(released|aired|founded|published|created|written)\b", text
    ):
        return "year"
    return None


def year_question(entity: str, kind: str, question: str) -> str:
    text = question.lower()
    if kind == "born":
        return f"In what year was {entity} born?"
    if kind == "died":
        return f"In what year did {entity} die?"
    if "founded" in text:
        return f"In what year was {entity} founded?"
    if "aired" in text:
        return f"In what year was {entity} first aired?"
    if "published" in text or "written" in text:
        return f"In what year was {entity} published?"
    if "created" in text:
        return f"In what year was {entity} created?"
    return f"In what year was {entity} released?"


def plan(question: str) -> dict | None:
    pair = split_pair(question)
    kind = attribute(question)
    later = direction_later(question)
    if pair is None or kind is None or later is None:
        return None
    return {"left": pair[0], "right": pair[1], "kind": kind, "later": later}


def focus_passages(entity: str, pages: list[Passage], limit: int = 4) -> list[Passage]:
    ranked: list[tuple[int, int, Passage]] = []
    needle = entity.lower()
    order = 0
    for page in pages:
        title_hit = needle in page.title.lower()
        for sentence in split_sentences(page.text):
            if needle not in sentence.lower() and not title_hit:
                continue
            year_hit = 1 if YEAR.search(sentence) else 0
            snippet = Passage(
                pid=page.pid,
                title=page.title,
                text=sentence,
                gold=page.gold,
                year=page.year,
                source_id=page.source_id,
            )
            ranked.append((year_hit, -order, snippet))
            order += 1
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = []
    seen = set()
    for _year_hit, _order, snippet in ranked:
        if snippet.text in seen:
            continue
        seen.add(snippet.text)
        chosen.append(snippet)
        if len(chosen) >= limit:
            break
    return chosen


def hop_prompt(question: str, entity: str, kind: str, pages: list[Passage]) -> str | None:
    evidence = focus_passages(entity, pages)
    if not evidence:
        return None
    asked = year_question(entity, kind, question)
    return evidence_prompt(asked, evidence)


def director_prompt(entity: str, pages: list[Passage]) -> str | None:
    evidence = focus_passages(entity, pages)
    if not evidence:
        return None
    return evidence_prompt(f"Who directed {entity}?", evidence)


def parse_year(text: str) -> int | None:
    match = YEAR.search(text or "")
    if match is None:
        return None
    return int(match.group(0))


def parse_person(raw: str) -> str:
    answer, _quote = parse_answer(raw)
    answer = " ".join(answer.split())
    if not answer or answer.lower() in {"unknown", "none", "n/a"}:
        return ""
    if len(answer.split()) > 12:
        return ""
    return answer


def decide(left: str, right: str, left_year: int | None, right_year: int | None, later: bool) -> str:
    if left_year is None or right_year is None or left_year == right_year:
        return ""
    if later:
        return left if left_year > right_year else right
    return left if left_year < right_year else right


def executed_answers(generate, count, items: list[tuple[str, list]]) -> list[dict]:
    """Run the frozen comparison. ``generate`` maps prompts to raw strings.

    A missing year leaves the answer empty so the caller keeps the witness reader.
    """
    results = [
        {"answer": "", "decision": "witness", "hop_tokens": 0}
        for _item in items
    ]
    active = []
    for index, (question, pages) in enumerate(items):
        spec = plan(question)
        if spec is not None:
            active.append((index, question, pages, spec))
    if not active:
        return results
    prompts = []
    meta = []
    for index, question, pages, spec in active:
        for side in ("left", "right"):
            entity = spec[side]
            if spec["kind"].startswith("director"):
                prompt = director_prompt(entity, pages)
                role = "director"
            else:
                prompt = hop_prompt(question, entity, spec["kind"], pages)
                role = "year"
            prompts.append(prompt or "")
            meta.append((index, side, role, prompt))
    nonempty = [prompt for prompt in prompts if prompt]
    raw = generate(nonempty) if nonempty else []
    cursor = 0
    people: dict[tuple[int, str], str] = {}
    years: dict[tuple[int, str], int | None] = {}
    hop_tokens: dict[int, int] = {}
    for index, side, role, prompt in meta:
        if not prompt:
            continue
        text = raw[cursor]
        cursor += 1
        hop_tokens[index] = hop_tokens.get(index, 0) + count(prompt)
        if role == "director":
            people[(index, side)] = parse_person(text)
        else:
            years[(index, side)] = parse_year(text)
    second_prompts = []
    second_meta = []
    for index, question, pages, spec in active:
        if not spec["kind"].startswith("director"):
            continue
        year_kind = "born" if spec["kind"].endswith("born") else "died"
        for side in ("left", "right"):
            person = people.get((index, side), "")
            prompt = hop_prompt(question, person, year_kind, pages) if person else None
            second_prompts.append(prompt or "")
            second_meta.append((index, side, prompt))
    nonempty = [prompt for prompt in second_prompts if prompt]
    raw_second = generate(nonempty) if nonempty else []
    cursor = 0
    for index, side, prompt in second_meta:
        if not prompt:
            continue
        years[(index, side)] = parse_year(raw_second[cursor])
        cursor += 1
        hop_tokens[index] = hop_tokens.get(index, 0) + count(prompt)
    for index, _question, _pages, spec in active:
        structural = decide(
            spec["left"],
            spec["right"],
            years.get((index, "left")),
            years.get((index, "right")),
            spec["later"],
        )
        results[index] = {
            "answer": structural,
            "decision": "execute" if structural else "witness",
            "hop_tokens": hop_tokens.get(index, 0),
        }
    return results


def _self_check() -> None:
    from epirag.data import Passage

    pages = [
        Passage("p0", "Ann", "Ann was born in 1980 in Paris and lived there.", False, 2020, "s0"),
        Passage("p1", "Bea", "Bea was born in 1990 in Paris and lived there.", False, 2020, "s1"),
    ]
    question = "Who was born first, Ann or Bea?"

    def generate(prompts: list[str]) -> list[str]:
        return ["1980" if "Ann" in prompt else "1990" for prompt in prompts]

    got = executed_answers(generate, len, [(question, pages), ("What is the capital of France?", pages)])
    assert got[0]["answer"] == "Ann" and got[0]["decision"] == "execute", got[0]
    assert got[1]["decision"] == "witness" and got[1]["answer"] == ""
    film = "Which film has the director who was born later, South Of Algiers or Kabut Sutra Ungu?"
    spec = plan(film)
    assert spec is not None and spec["kind"] == "director-born" and spec["later"] is True
    print("execute self-check passed")


if __name__ == "__main__":
    _self_check()
