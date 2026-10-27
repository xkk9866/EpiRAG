#!/bin/bash
cd "$(dirname "$0")/.."
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
PY=${PYTHON:-python}; O=experiments/synth; mkdir -p $O/logs
S="--encoder microsoft/deberta-v3-base --out $O --datasets synth2,synth3,synth4 --n-train 1500 --n-dev 100 --n-test 300 --epochs 3 --max-len 128 --batch 8 --accum 2 --workers 1 --k 8"
run() { n=$1; shift; $PY -m quotenet.train --name $n $S "$@" > $O/logs/$n.log 2>&1 && echo OK $n >> $O/done.txt || echo FAIL $n >> $O/done.txt; }
for s in 13 29 41; do
 run indep_s$s --mode indep --seed $s
 run belief_s$s --mode belief --seed $s
 run quote_s$s --mode quote --seed $s
 run quoteR2_s$s --mode quote --rounds 2 --seed $s
 run quoteR3_s$s --mode quote --rounds 3 --seed $s
 run slots_s$s --mode slots --seed $s
done
