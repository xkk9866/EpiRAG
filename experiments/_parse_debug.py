import re

_LABEL_FIND = re.compile(
    r"(?i)\b(Q1B|Q2B|E1|E2|Q1|Q2)\s*[:.\-]\s*(.*?)(?=\b(?:Q1B|Q2B|E1|E2|Q1|Q2)\s*[:.\-]|$)"
)
text = "CHAIN\nQ1: Who directed Film?\nQ2: Where did PERSON die?"
for match in _LABEL_FIND.finditer(text):
    print(repr(match.group(1)), repr(match.group(2)))
