"""Turn a relational question into one or two focused hops.

hop2 uses the literal token PERSON, which the caller replaces with the
first hop's answer. A missing decomposition leaves the witness reader in place.
"""

from __future__ import annotations

import re

_FRAMES: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"(?i)^where did the (?P<role>director|composer|performer) of (?P<kind>film|song|album) (?P<work>.+) die$"),
        "Who is the {role} of {work}?",
        "Where did PERSON die?",
    ),
    (
        re.compile(r"(?i)^where was the (?P<role>director|composer|performer) of (?P<kind>film|song|album) (?P<work>.+) born$"),
        "Who is the {role} of {work}?",
        "Where was PERSON born?",
    ),
    (
        re.compile(r"(?i)^what is the place of birth of the (?P<role>director|composer|performer) of (?P<kind>film|song|album) (?P<work>.+)$"),
        "Who is the {role} of {work}?",
        "Where was PERSON born?",
    ),
    (
        re.compile(r"(?i)^where was the place of death of the director of film (?P<work>.+)$"),
        "Who is the director of {work}?",
        "Where did PERSON die?",
    ),
    (
        re.compile(r"(?i)^what nationality is the director of film (?P<work>.+)$"),
        "Who is the director of {work}?",
        "What is the nationality of PERSON?",
    ),
    (
        re.compile(r"(?i)^which country the director of film (?P<work>.+) is from$"),
        "Who is the director of {work}?",
        "Which country is PERSON from?",
    ),
    (
        re.compile(r"(?i)^who is the mother of the composer of film (?P<work>.+)$"),
        "Who is the composer of {work}?",
        "Who is PERSON's mother?",
    ),
    (
        re.compile(r"(?i)^where does the director of film (?P<work>.+) work(?: at)?$"),
        "Who is the director of {work}?",
        "Where does PERSON work?",
    ),
    (
        re.compile(r"(?i)^where did the director of film (?P<work>.+) graduate from$"),
        "Who is the director of {work}?",
        "Where did PERSON graduate from?",
    ),
    (
        re.compile(r"(?i)^which award the performer of song (?P<work>.+)$"),
        "Who is the performer of {work}?",
        "Which award did PERSON receive?",
    ),
    (
        re.compile(r"(?i)^what is the place of birth of the performer of song (?P<work>.+)$"),
        "Who is the performer of {work}?",
        "Where was PERSON born?",
    ),
    (
        re.compile(r"(?i)^where did the composer of song (?P<work>.+) die$"),
        "Who is the composer of {work}?",
        "Where did PERSON die?",
    ),
    (
        re.compile(r"(?i)^who is the founder of the company which published (?P<work>.+)$"),
        "Which company published {work}?",
        "Who founded PERSON?",
    ),
    (
        re.compile(r"(?i)^when did (?P<name>.+)'s (?P<rel>father|mother|husband|wife) die$"),
        "Who is {name}'s {rel}?",
        "When did PERSON die?",
    ),
    (
        re.compile(r"(?i)^where did (?P<name>.+)'s (?P<rel>father|mother|husband|wife) die$"),
        "Who is {name}'s {rel}?",
        "Where did PERSON die?",
    ),
    (
        re.compile(r"(?i)^what nationality is (?P<name>.+)'s (?P<rel>father|mother)$"),
        "Who is {name}'s {rel}?",
        "What is the nationality of PERSON?",
    ),
    (
        re.compile(r"(?i)^what is the place of birth of (?P<name>.+)'s (?P<rel>father|mother|husband|wife)$"),
        "Who is {name}'s {rel}?",
        "Where was PERSON born?",
    ),
    (
        re.compile(r"(?i)^where was the place of death of (?P<name>.+)'s (?P<rel>father|mother|husband|wife)$"),
        "Who is {name}'s {rel}?",
        "Where did PERSON die?",
    ),
    (
        re.compile(r"(?i)^where was the place of burial of (?P<name>.+)'s (?P<rel>father|mother|husband|wife)$"),
        "Who is {name}'s {rel}?",
        "Where was PERSON buried?",
    ),
    (
        re.compile(r"(?i)^which country the (?P<role>director|composer|performer) of (?P<kind>film|song|album) (?P<work>.+) is from$"),
        "Who is the {role} of {work}?",
        "Which country is PERSON from?",
    ),
    (
        re.compile(r"(?i)^what is the award that the (?P<role>director|performer) of (?P<kind>film|song) (?P<work>.+) received$"),
        "Who is the {role} of {work}?",
        "Which award did PERSON receive?",
    ),
    (
        re.compile(r"(?i)^who is the spouse of the director of film (?P<work>.+)$"),
        "Who is the director of {work}?",
        "Who is PERSON's spouse?",
    ),
    (
        re.compile(r"(?i)^what is the date of birth of (?P<name>.+)'s (?P<rel>father|mother)$"),
        "Who is {name}'s {rel}?",
        "When was PERSON born?",
    ),
    (
        re.compile(r"(?i)^where does (?P<name>.+)'s (?P<rel>father|mother) work(?: at)?$"),
        "Who is {name}'s {rel}?",
        "Where does PERSON work?",
    ),
    (
        re.compile(r"(?i)^where was the (?P<rel>father|mother) of (?P<name>.+) born$"),
        "Who is {name}'s {rel}?",
        "Where was PERSON born?",
    ),
    (
        re.compile(r"(?i)^why did the director of film (?P<work>.+) die$"),
        "Who is the director of {work}?",
        "Why did PERSON die?",
    ),
]


def _fill(template: str, groups: dict[str, str]) -> str:
    text = template
    for key, value in groups.items():
        text = text.replace("{" + key + "}", value.strip())
    return " ".join(text.split())


def decompose(question: str) -> dict | None:
    text = " ".join((question or "").strip().rstrip("?").split())
    grand = re.match(
        r"(?i)^who is (?P<name>.+?)'s (?P<side>maternal|paternal) (?P<rel>grandfather|grandmother)$",
        text,
    )
    of_grand = re.match(
        r"(?i)^who is the (?P<side>maternal|paternal) (?P<rel>grandfather|grandmother) of (?P<name>.+)$",
        text,
    )
    matched = grand or of_grand
    if matched:
        side = matched.group("side").lower()
        rel = matched.group("rel").lower()
        name = matched.group("name").strip()
        parent = "mother" if side == "maternal" else "father"
        elder = "father" if rel == "grandfather" else "mother"
        return {
            "hop1": f"Who is {name}'s {parent}?",
            "hop2": f"Who is PERSON's {elder}?",
            "focus": name,
        }
    for pattern, hop1, hop2 in _FRAMES:
        found = pattern.match(text)
        if not found:
            continue
        groups = {key: value for key, value in found.groupdict().items() if value}
        focus = groups.get("work") or groups.get("name") or ""
        return {"hop1": _fill(hop1, groups), "hop2": _fill(hop2, groups), "focus": focus}
    direct = re.match(r"(?i)^who is (?P<name>.+?)'s (?P<rel>mother|father|husband|wife|child)$", text)
    of_direct = re.match(r"(?i)^who is the (?P<rel>mother|father|uncle) of (?P<name>.+)$", text)
    one = direct or of_direct
    if one:
        name = one.group("name").strip()
        return {"hop1": f"Who is {name}'s {one.group('rel').lower()}?", "hop2": "", "focus": name}
    return None


_REJECT = {"", "unknown", "none", "n/a", "unanswerable"}


def _acceptable(answer: str) -> bool:
    text = " ".join((answer or "").split())
    key = text.lower().strip(" .")
    if key in _REJECT or key.startswith("no information"):
        return False
    if len(text.split()) > 12:
        return False
    return True


def resolve_hops(generate, items: list[tuple[str, list]], count=None) -> list[dict]:
    """Return a structural span, or an empty answer when the witness reader should answer.

    A hop answer is kept only when that span is stated in the sentences shown
    to the hop. No cutoff is fit after a slice is scored.
    """
    from epirag.engine import evidence_prompt, parse_answer
    from epirag.execute import focus_passages
    from epirag.metrics import normalize_answer

    def stated(answer: str, pages) -> bool:
        gold = normalize_answer(answer).split()
        if not gold:
            return False
        have: set[str] = set()
        for page in pages:
            have.update(normalize_answer(page.text).split())
        return sum(token in have for token in gold) / len(gold) >= 0.5

    def blank() -> dict:
        return {"answer": "", "hop_tokens": 0}

    results = [blank() for _item in items]
    active = []
    for index, (question, pages) in enumerate(items):
        spec = decompose(question)
        if spec is not None:
            active.append((index, pages, spec))
    if not active:
        return results
    tokens: dict[int, int] = {}
    first_prompts = []
    first_meta = []
    for index, pages, spec in active:
        evidence = focus_passages(spec["focus"], pages)
        prompt = evidence_prompt(spec["hop1"], evidence) if evidence else None
        first_prompts.append(prompt or "")
        first_meta.append((index, prompt, evidence, spec))
    raw = generate([prompt for prompt in first_prompts if prompt]) if any(first_prompts) else []
    cursor = 0
    second_prompts = []
    second_meta = []
    for index, prompt, evidence, spec in first_meta:
        if not prompt:
            continue
        if count is not None:
            tokens[index] = tokens.get(index, 0) + count(prompt)
        answer, _quote = parse_answer(raw[cursor])
        cursor += 1
        answer = " ".join(answer.split())
        if not (_acceptable(answer) and stated(answer, evidence)):
            continue
        if not spec["hop2"]:
            results[index] = {"answer": answer, "hop_tokens": tokens.get(index, 0)}
            continue
        asked = spec["hop2"].replace("PERSON", answer)
        pages = next(pages for item_index, pages, _spec in active if item_index == index)
        second_evidence = focus_passages(answer, pages)
        second = evidence_prompt(asked, second_evidence) if second_evidence else None
        second_prompts.append(second or "")
        second_meta.append((index, second, second_evidence))
    raw2 = generate([prompt for prompt in second_prompts if prompt]) if any(second_prompts) else []
    cursor = 0
    for index, prompt, evidence in second_meta:
        if not prompt:
            continue
        if count is not None:
            tokens[index] = tokens.get(index, 0) + count(prompt)
        answer, _quote = parse_answer(raw2[cursor])
        cursor += 1
        answer = " ".join(answer.split())
        if _acceptable(answer) and stated(answer, evidence):
            results[index] = {"answer": answer, "hop_tokens": tokens.get(index, 0)}
        else:
            results[index] = {"answer": "", "hop_tokens": tokens.get(index, 0)}
    for index, _pages, _spec in active:
        results[index]["hop_tokens"] = tokens.get(index, results[index]["hop_tokens"])
    return results

