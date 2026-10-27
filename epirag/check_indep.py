"""CPU checks for independent corroboration. No generator and no NLI weights."""

from epirag.data import Passage
from epirag.engine import coordinate


class TableNLI:
    def __init__(self, hits):
        self.hits = hits

    def score(self, pairs):
        rows = []
        for sentence, hypothesis in pairs:
            matched = None
            for claim, sentences in self.hits.items():
                if claim in hypothesis and sentence in sentences:
                    matched = sentences[sentence]
                    break
            rows.append(matched or (0.05, 0.02, 0.93))
        return rows


def page(pid, source, text):
    return Passage(pid, pid, text, True, 2020, source)


PARIS = "Paris is the capital city of France according to this page."
LYON = "Lyon is the capital city of France according to this page."
WITNESS_PARIS = "An independent page states that Paris is the capital of France."
WITNESS_BOTH = "An independent page names both Paris and Lyon as candidate capitals."
NEUTRAL = "The river discussed here is unrelated to either proposed capital."


def run_case(name, answers, hits, mode, third, allow_conflict=False, archive=None):
    agents = [
        [page("a", "s0", PARIS)],
        [page("b", "s1", LYON)],
        [page("c", "s2", third)],
    ]
    state = coordinate(
        "What is the capital of France?",
        archive or [],
        agents,
        answers,
        TableNLI(hits),
        mode=mode,
        allow_retrieval=bool(archive),
        allow_conflict=allow_conflict,
    )
    print(
        name,
        "decision",
        state["decision"],
        "structural",
        state["structural"],
        "triggered",
        state["triggered"],
        "beliefs",
        [round(v, 3) for v in state["beliefs"]],
        "nli",
        state["nli_pairs"],
    )
    return state


def main():
    external = run_case(
        "one-external",
        ["Paris", "Lyon", "unknown"],
        {"Paris": {WITNESS_PARIS: (0.02, 0.91, 0.07), PARIS: (0.02, 0.80, 0.18)}},
        "indep",
        WITNESS_PARIS,
    )
    assert external["decision"] == "corroborated"
    assert external["structural"] == "Paris"
    assert all(value == 1.0 for value in external["reliability"].values())

    both = run_case(
        "two-external",
        ["Paris", "Lyon", ""],
        {
            "Paris": {WITNESS_BOTH: (0.02, 0.91, 0.07)},
            "Lyon": {WITNESS_BOTH: (0.02, 0.87, 0.11)},
        },
        "indep",
        WITNESS_BOTH,
    )
    assert both["structural"] == ""
    assert both["decision"] == "reader"

    conflict = run_case(
        "two-conflict",
        ["Paris", "Lyon", ""],
        {
            "Paris": {WITNESS_BOTH: (0.02, 0.91, 0.07)},
            "Lyon": {WITNESS_BOTH: (0.02, 0.87, 0.11)},
        },
        "indep",
        WITNESS_BOTH,
        allow_conflict=True,
    )
    assert conflict["structural"] == "conflict"
    assert conflict["decision"] == "conflict"

    self_only = run_case(
        "self-conflict",
        ["Paris", "Lyon", ""],
        {"Paris": {PARIS: (0.02, 0.90, 0.08)}, "Lyon": {LYON: (0.02, 0.90, 0.08)}},
        "indep",
        NEUTRAL,
        allow_conflict=True,
    )
    assert self_only["structural"] == "conflict"

    seek = run_case(
        "seek-reader",
        ["Paris", "Lyon", ""],
        {},
        "indep_never",
        NEUTRAL,
    )
    assert seek["decision"] == "reader"
    assert seek["structural"] == ""
    assert seek["triggered"] is False

    graph = run_case(
        "graph",
        ["Paris", "Lyon", ""],
        {"Paris": {PARIS: (0.80, 0.10, 0.10)}},
        "graph",
        NEUTRAL,
    )
    assert graph["decision"] == "reader"
    assert graph["structural"] == ""

    daughter = "Elizabeth Bruce was the daughter of Robert the Bruce in this chronicle."
    born = "Robert the Bruce was born in Scotland and ruled from that country."
    agents_bridge = [
        [page("a", "s0", daughter)],
        [page("b", "s1", born)],
        [page("c", "s2", NEUTRAL)],
    ]
    bridged = coordinate(
        "Where was the father of Elizabeth Bruce born?",
        [],
        agents_bridge,
        ["Robert the Bruce", "Scotland", ""],
        TableNLI(
            {
                "Scotland": {born: (0.02, 0.92, 0.06)},
                "Robert the Bruce": {daughter: (0.02, 0.90, 0.08)},
            }
        ),
        mode="bridge",
        allow_retrieval=False,
    )
    print("bridge", bridged["decision"], bridged["structural"])
    assert bridged["decision"] == "bridge"
    assert bridged["structural"] == "Scotland"

    quiet = coordinate(
        "What is the capital of France?",
        [],
        [
            [page("a", "s0", PARIS)],
            [page("b", "s1", LYON)],
            [page("c", "s2", NEUTRAL)],
        ],
        ["Paris", "Lyon", ""],
        TableNLI(
            {
                "Paris": {PARIS: (0.05, 0.90, 0.05)},
                "Lyon": {LYON: (0.05, 0.80, 0.15)},
            }
        ),
        mode="gate",
        allow_retrieval=False,
    )
    print("gate-pool", quiet["decision"], quiet["mode"])
    assert quiet["decision"] == "pool"
    assert quiet["structural"] == ""

    fought = coordinate(
        "What is the capital of France?",
        [],
        [
            [page("a", "s0", PARIS)],
            [page("b", "s1", LYON)],
            [page("c", "s2", NEUTRAL)],
        ],
        ["Paris", "Lyon", ""],
        TableNLI(
            {
                "Paris": {PARIS: (0.01, 0.99, 0.00), LYON: (0.80, 0.05, 0.15)},
                "Lyon": {LYON: (0.05, 0.20, 0.75)},
            }
        ),
        mode="gate",
        allow_retrieval=False,
    )
    print("gate-witness", fought["decision"], fought["mode"])
    assert fought["decision"] == "witness"
    assert fought["structural"] == ""

    labeled = coordinate(
        "What is the capital of France?",
        [],
        [
            [page("a", "s0", PARIS)],
            [page("b", "s1", LYON)],
            [page("c", "s2", NEUTRAL)],
        ],
        ["Paris", "Lyon", ""],
        TableNLI(
            {
                "Paris": {PARIS: (0.01, 0.95, 0.04), LYON: (0.90, 0.04, 0.06)},
                "Lyon": {PARIS: (0.88, 0.04, 0.08), LYON: (0.02, 0.55, 0.43)},
            }
        ),
        mode="gate",
        allow_retrieval=False,
        allow_conflict=True,
    )
    print("gate-conflict", labeled["decision"], labeled["structural"])
    assert labeled["structural"] == "conflict"

    marked = coordinate(
        "What is the capital of France?",
        [],
        [
            [page("a", "s0", PARIS)],
            [page("b", "s1", LYON)],
            [page("c", "s2", NEUTRAL)],
        ],
        ["Paris", "Lyon", ""],
        TableNLI(
            {
                "Paris": {PARIS: (0.01, 0.99, 0.00), LYON: (0.80, 0.05, 0.15)},
                "Lyon": {LYON: (0.05, 0.20, 0.75)},
            }
        ),
        mode="mark",
        allow_retrieval=False,
    )
    print("mark-conflict", marked["decision"], marked["tags"].get(LYON))
    assert marked["decision"] == "marked"
    assert marked["structural"] == ""
    assert "CONTRADICTS" in marked["tags"][LYON]
    assert "SUPPORTS" in marked["tags"][PARIS]

    quiet_mark = coordinate(
        "What is the capital of France?",
        [],
        [[page("a", "s0", PARIS)], [page("b", "s1", PARIS)], [page("c", "s2", NEUTRAL)]],
        ["Paris", "Paris", "Paris"],
        TableNLI({"Paris": {PARIS: (0.01, 0.93, 0.06), NEUTRAL: (0.05, 0.05, 0.90)}}),
        mode="mark",
        allow_retrieval=False,
    )
    print("mark-plain", quiet_mark["decision"])
    assert quiet_mark["decision"] == "plain"
    assert quiet_mark["tags"] == {}
    print("ok")


if __name__ == "__main__":
    main()
