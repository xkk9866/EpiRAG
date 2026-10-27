#!/bin/bash
# usage: run_local.sh name [extra args]
cd "$(dirname "$0")/../.."
mkdir -p experiments/relay/logs
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} PYTHONWARNINGS=ignore
n=$1; shift
python -m epinet.train_relay --name $n --n-train 6000 --n-dev 150 --n-test 500 --epochs 2 --batch 2 --accum 4 --out experiments/relay "$@" > experiments/relay/logs/$n.log 2>&1 && echo OK $n || echo FAIL $n
