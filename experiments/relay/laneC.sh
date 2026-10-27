#!/bin/bash
# counterfactual probes once each run has finished testing
cd "$(dirname "$0")/../.."
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} PYTHONWARNINGS=ignore
for spec in "relay_s13 relay 1" "belief_s13 belief 1" "intra_s13 intra 1" "late_s13 late 1" "relay_s29 relay 1" "relay_s41 relay 1"; do
  set -- $spec
  until [ -f experiments/relay/$1/test_summary.json ]; do sleep 120; done
  for pr in swap drop; do
    ${PYTHON:-python} -m epinet.train_relay --name $1 --mode $2 --encoder microsoft/deberta-v3-base --n-dev 150 --n-test 500 \
      --out experiments/relay --eval-only --probe $pr > experiments/relay/logs/probe_${1}_$pr.log 2>&1
    echo probe $1 $pr $(date)
  done
done
echo LANEC_DONE $(date)
