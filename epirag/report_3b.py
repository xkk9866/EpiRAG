from pathlib import Path

from epirag.paired_tests import report

report(
    Path("experiments/results/records_confirm_3b.jsonl"),
    [
        ("epirag_cover", "epirag_flat"),
        ("epirag_cover", "debate"),
        ("epirag_cover", "centralized"),
        ("epirag_cover", "epirag_drop"),
        ("epirag_cover", "epirag_cover_noret"),
    ],
)
