"""SQuAD-style answer metrics and calibration."""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction).split()
    gold_toks = normalize_answer(gold).split()
    if not pred and not gold_toks:
        return 1.0
    if not pred or not gold_toks:
        return 0.0
    common = Counter(pred) & Counter(gold_toks)
    num = sum(common.values())
    if num == 0:
        return 0.0
    precision = num / len(pred)
    recall = num / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def expected_calibration_error(
    confidences: list[float], correct: list[float], n_bins: int = 10
) -> float:
    if not confidences:
        return 0.0
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    corr = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    ece = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf > edges[i]) & (conf <= edges[i + 1])
        count = int(mask.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(float(corr[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def brier_score(confidences: list[float], correct: list[float]) -> float:
    if not confidences:
        return 0.0
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    corr = np.asarray(correct, dtype=float)
    return float(np.mean((conf - corr) ** 2))
