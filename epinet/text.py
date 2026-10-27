"""Text normalisation, answer metrics, sentence splitting and value parsing."""

from __future__ import annotations

import hashlib
import math
import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_TOKEN = re.compile(r"[a-z0-9]+")
_PUNCT = set(string.punctuation)

REFUSALS = {
    "", "unknown", "none", "n a", "na", "not mentioned", "not stated", "no answer",
    "cannot be determined", "unanswerable", "not enough information", "insufficient information",
    "no information", "not specified", "not provided", "not given", "i dont know",
}

MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def tokenize(text: str | None) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(gold).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_em(prediction: str, golds: list[str]) -> float:
    return max((exact_match(prediction, g) for g in golds), default=0.0)


def best_f1(prediction: str, golds: list[str]) -> float:
    return max((token_f1(prediction, g) for g in golds), default=0.0)


def is_refusal(text: str | None) -> bool:
    key = normalize_answer(text)
    if key in REFUSALS:
        return True
    return key.startswith(("no information", "the evidence does not", "evidence does not", "not mentioned"))


def canonical_polar(question: str, answer: str) -> str:
    """Map true/false style replies onto the yes/no lexicon polar questions use."""
    key = normalize_answer(answer)
    if not is_polar(question):
        return (answer or "").strip()
    if key in {"yes", "true", "correct"} or key.startswith("yes "):
        return "yes"
    if key in {"no", "false", "incorrect"} or key.startswith("no "):
        return "no"
    return (answer or "").strip()


_POLAR = ("are ", "is ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ", "will ")


def is_polar(question: str) -> bool:
    return (question or "").strip().lower().startswith(_POLAR)


def split_sentences(text: str, min_chars: int = 12) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", (text or "").strip())
    kept = [p.strip() for p in parts if len(p.strip()) >= min_chars]
    if not kept and (text or "").strip():
        kept = [text.strip()]
    return kept


def stated_ratio(sentence: str, claim: str) -> float:
    """Fraction of the claim's tokens that occur in the sentence."""
    toks = tokenize(claim)
    if not toks:
        return 0.0
    sent = set(tokenize(sentence))
    return sum(t in sent for t in toks) / len(toks)


_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_DAY_MONTH_YEAR = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{3,4})\b")
_MONTH_DAY_YEAR = re.compile(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{3,4})\b")
_MONTH_YEAR = re.compile(r"\b([A-Za-z]+)\s+(\d{3,4})\b")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_value(text: str | None) -> float:
    """Return a comparable scalar for a date, a year or a number, else NaN.

    Dates map to fractional years so that day-level comparisons stay ordered.
    """
    raw = (text or "").strip()
    if not raw:
        return math.nan
    for pattern, order in ((_DAY_MONTH_YEAR, "dmy"), (_MONTH_DAY_YEAR, "mdy")):
        match = pattern.search(raw)
        if match:
            if order == "dmy":
                day, month, year = match.group(1), match.group(2), match.group(3)
            else:
                month, day, year = match.group(1), match.group(2), match.group(3)
            month_id = MONTHS.get(month.lower())
            if month_id:
                return float(year) + (month_id - 1) / 12.0 + (int(day) - 1) / 365.0
    match = _MONTH_YEAR.search(raw)
    if match and match.group(1).lower() in MONTHS:
        return float(match.group(2)) + (MONTHS[match.group(1).lower()] - 1) / 12.0
    match = _YEAR.search(raw)
    if match:
        return float(match.group(1))
    match = _NUMBER.search(raw)
    if match:
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return math.nan
    return math.nan


def same_answer(left: str, right: str) -> bool:
    """Conservative string identity used to merge agent claims."""
    a, b = normalize_answer(left), normalize_answer(right)
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = a.split(), b.split()
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    # "Paris" vs "Paris France" merges; "John Smith" vs "John Brown" does not.
    if len(short) >= 1 and all(t in long_ for t in short) and token_f1(a, b) >= 0.66:
        return True
    return False
