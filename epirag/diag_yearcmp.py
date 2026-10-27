"""Development: compare two question entities by a year stated in the pages.

Abstain unless both sides have a year. Fallback is the stored witness answer.
"""

import json
import re
from collections import defaultdict

from datasets import load_dataset

from epirag.data import build_instance, select_examples
from epirag.diag_chainsearch import pages_of, sentences_of
from epirag.diag_frontier import mentions
from epirag.engine import canonicalize_polar
from epirag.metrics import exact_match

STRESS = ("asymmetry", "corruption", "temporal", "incomplete")
FILES = {
    "advance": "experiments/results/records_advance.jsonl",
    "forward": "experiments/results/records_forward.jsonl",
}
YEAR = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")
LATER = re.compile(r"\b(later|after|younger|second)\b", re.I)
EARLIER = re.compile(r"\b(earlier|first|before|older|born first|aired first|came out earlier|released earlier)\b", re.I)


def pair(question: str) -> tuple[str, str] | None:
    if not EARLIER.search(question) and not LATER.search(question):
        return None
    found = []
    for mention in mentions(question):
        if mention.lower() in {"which", "who", "what", "when"}:
            continue
        if mention not in found:
            found.append(mention)
    if len(found) < 2:
        return None
    # The two compared names are the last two capitalized phrases in these templates.
    return found[-2], found[-1]


def years_for(entity: str, question: str, rows) -> list[int]:
    born_question = bool(re.search(r"\bborn\b", question.lower()))
    direct = []
    births = []
    people = []
    for text, _source in rows:
        if f" {entity.lower()} " not in f" {text.lower()} ":
            continue
        yrs = [int(item) for item in YEAR.findall(text)]
        if yrs:
            direct.extend(yrs)
        if born_question:
            for person in mentions(text):
                if person.lower() == entity.lower():
                    continue
                people.append(person)
    if born_question:
        for person in people:
            for text, _source in rows:
                if person.lower() not in text.lower() or not re.search(r"\bborn\b", text.lower()):
                    continue
                births.extend(int(item) for item in YEAR.findall(text))
        if births:
            return births
    return direct


def choose(question: str, pages) -> str:
    compared = pair(question)
    if compared is None:
        return ""
    left, right = compared
    rows = sentences_of(pages)
    left_years = years_for(left, question, rows)
    right_years = years_for(right, question, rows)
    if not left_years or not right_years:
        return ""
    # A corrupt copy can add a second year. Use the median so one planted year does not decide.
    left_year = sorted(left_years)[len(left_years) // 2]
    right_year = sorted(right_years)[len(right_years) // 2]
    if left_year == right_year:
        return ""
    later = bool(LATER.search(question)) and not bool(EARLIER.search(question))
    if later:
        return left if left_year > right_year else right
    return left if left_year < right_year else right


def main():
    stored = defaultdict(dict)
    for split, path in FILES.items():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["condition"] not in STRESS or row["method"] != "epirag_cover":
                    continue
                stored[split][(row["dataset"], row["qid"], row["condition"])] = row
    hotpot = load_dataset("hotpot_qa", "distractor", split="validation")
    wiki = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    specs = (("hotpotqa", hotpot, 200), ("2wiki", wiki, 160))
    for split in FILES:
        print("==", split, flush=True)
        for name, data, budget in specs:
            _dev, ids = select_examples(data, name, budget, n_dev=12, seed=7, block=split)
            instances = [build_instance(data[i], name) for i in ids]
            n = cover_hit = hybrid_hit = fired = fire_hit = 0
            for instance in instances:
                for condition in STRESS:
                    row = stored[split].get((name, instance.qid, condition))
                    if row is None:
                        continue
                    n += 1
                    cover = canonicalize_polar(instance.question, row["answer"])
                    structural = choose(instance.question, pages_of(instance, condition))
                    pred = structural or cover
                    if structural:
                        fired += 1
                        fire_hit += exact_match(structural, instance.answer)
                    cover_hit += exact_match(cover, instance.answer)
                    hybrid_hit += exact_match(pred, instance.answer)
            prec = 100 * fire_hit / fired if fired else 0.0
            print(
                f"{name} n {n} cover {100 * cover_hit / n:.2f} hybrid {100 * hybrid_hit / n:.2f} "
                f"fire {100 * fired / n:.1f} precision {prec:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
