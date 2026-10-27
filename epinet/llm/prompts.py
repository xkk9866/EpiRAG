"""Prompts shared by EpiNet and all baselines.

Every method uses the same agent prompt for a local answer, so differences
between methods come from what crosses the view boundary and how it is
decided, not from prompt wording.
"""

from __future__ import annotations

import re

from epinet.data.examples import Page
from epinet.text import is_polar


def render_pages(pages: list[Page], show_year: bool = True) -> str:
    blocks = []
    for i, page in enumerate(pages, start=1):
        head = f"[{i}] {page.title}" + (f" (dated {page.year})" if show_year else "")
        blocks.append(head + "\n" + page.text)
    return "\n".join(blocks) if blocks else "[1] (no evidence)"


def render_sentences(items: list[tuple[str, str]]) -> str:
    """``items`` are (title, sentence) pairs."""
    if not items:
        return "- (no evidence)"
    return "\n".join(f"- ({title}) {sentence}" for title, sentence in items)


def agent_prompt(question: str, pages: list[Page]) -> str:
    polar = "If it is a yes/no question, answer yes or no.\n" if is_polar(question) else ""
    return (
        "Answer the question using only the evidence below. "
        "If the evidence does not contain the answer, write unknown.\n"
        f"{polar}"
        "Reply in exactly two lines:\n"
        "Answer: <short answer span>\n"
        "Quote: <the one sentence from the evidence that states the answer>\n\n"
        f"Question: {question}\n"
        f"Evidence:\n{render_pages(pages)}"
    )


def reader_prompt(question: str, pages: list[Page]) -> str:
    """Plain short-answer reading, used by the centralized pool and by debate."""
    polar = "If it is a yes/no question, answer yes or no.\n" if is_polar(question) else ""
    return (
        "Answer the question using only the evidence. "
        "Reply with the short answer span only. Do not explain.\n"
        f"{polar}\n"
        f"Question: {question}\n"
        f"Evidence:\n{render_pages(pages)}"
    )


def board_prompt(question: str, items: list[tuple[str, str]]) -> str:
    """The coordinator reads the message board: sentences agents chose to send."""
    polar = "If it is a yes/no question, answer yes or no.\n" if is_polar(question) else ""
    return (
        "Several agents shared the sentences below from their private documents. "
        "Some sentences may be outdated or wrong. Combine them to answer the question. "
        "Reply with the short answer span only. Do not explain.\n"
        f"{polar}\n"
        f"Question: {question}\n"
        f"Shared sentences:\n{render_sentences(items)}"
    )


PLAN_PROMPT = """Rewrite the question as lookups. Each lookup must be answerable from one encyclopedia page. Use #1, #2 to refer to the answer of an earlier lookup. Use one of the TYPE values: direct, chain, earlier, later, less, more, same, both. Output only the plan.

Question: Where did the director of film Blind Shaft die?
TYPE: chain
S1: Who directed Blind Shaft?
S2: Where did #1 die?

Question: Who is the maternal grandfather of Prince Ernst?
TYPE: chain
S1: Who is the mother of Prince Ernst?
S2: Who is the father of #1?

Question: Which film came out first, Blind Shaft or The Mask of Fu Manchu?
TYPE: earlier
E1: Blind Shaft
S1: When was Blind Shaft released?
E2: The Mask of Fu Manchu
S2: When was The Mask of Fu Manchu released?

Question: Which film has the director who was born later, Wild Horses or Dead Calm?
TYPE: later
E1: Wild Horses
S1: Who directed Wild Horses?
S1b: When was #1 born?
E2: Dead Calm
S2: Who directed Dead Calm?
S2b: When was #1 born?

Question: Which city has a larger population, Bergen or Uppsala?
TYPE: more
E1: Bergen
S1: What is the population of Bergen?
E2: Uppsala
S2: What is the population of Uppsala?

Question: Were Scott Derrickson and Ed Wood of the same nationality?
TYPE: same
S1: What is the nationality of Scott Derrickson?
S2: What is the nationality of Ed Wood?

Question: Are Vogue and Elle both fashion magazines?
TYPE: both
VALUE: fashion magazine
S1: What kind of publication is Vogue?
S2: What kind of publication is Elle?

Question: The arena where the Lewiston Maineiacs played can seat how many people?
TYPE: chain
S1: In which arena did the Lewiston Maineiacs play?
S2: How many people can #1 seat?

Question: What is the capital of the country where the Eiffel Tower is located?
TYPE: chain
S1: In which country is the Eiffel Tower located?
S2: What is the capital of #1?

Question: What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?
TYPE: direct

Question: {question}
"""


def plan_prompt(question: str) -> str:
    return PLAN_PROMPT.format(question=" ".join((question or "").split()))


_ANSWER = re.compile(r"(?im)^\s*\**answer\**\s*[:：]\s*(.+?)\s*$")
_QUOTE = re.compile(r"(?im)^\s*\**quote\**\s*[:：]\s*(.+?)\s*$")


def parse_agent(raw: str) -> tuple[str, str]:
    """Return (answer, quote). Tolerates missing labels and truncated output."""
    text = (raw or "").strip()
    answer_match = _ANSWER.search(text)
    quote_match = _QUOTE.search(text)
    if answer_match:
        answer = answer_match.group(1)
    else:
        first = text.splitlines()[0] if text else ""
        answer = re.sub(r"(?i)^answer\s*[:：]\s*", "", first)
    answer = answer.strip().strip('"').strip("*").strip()
    answer = re.sub(r"\s*\[\d+\]\s*$", "", answer).rstrip(".").strip()
    quote = quote_match.group(1).strip().strip('"') if quote_match else ""
    return answer[:120], quote[:400]


def parse_short(raw: str) -> str:
    text = (raw or "").strip()
    first = text.splitlines()[0] if text else ""
    first = re.sub(r"(?i)^(final\s+)?answer\s*[:：]\s*", "", first).strip().strip('"').strip("*")
    return first.rstrip(".").strip()[:120]
