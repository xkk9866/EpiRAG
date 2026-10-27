#!/bin/bash
# stage 1: shared independent reader per seed; stage 2: one epoch per communication variant, warm-started from it
cd "$(dirname "$0")/.."
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false
PY=${PYTHON:-python}; O=experiments/q2; mkdir -p $O/logs
run() { n=$1; shift; $PY -m quotenet.train --name $n --encoder microsoft/deberta-v3-base --out $O --workers 2 "$@" > $O/logs/$n.log 2>&1 && echo OK $n $(date) >> $O/done.txt || echo FAIL $n $(date) >> $O/done.txt; }
s1() { s=$1; [ -f $O/base_s$s/test_summary.json ] || run base_s$s --mode indep --seed $s --n-train 4000 --epochs 2; }
s2() { s=$1; v=$2; shift 2; run ${v}_s$s --seed $s --train-seed $((s+100)) --n-train 3000 --epochs 1 --lr 2e-5 --head-lr 2e-4 --init $O/base_s$s/model.pt "$@"; }
lane() { s=$1; shift; s1 $s; for spec in "$@"; do IFS=: read v rest <<< "$spec"; s2 $s $v $rest; done; }
case $1 in
 A) lane 13 "quote:--mode quote --k 32" "sg:--mode sg --k 32";;
 B) while [ ! -f $O/base_s13/test_summary.json ]; do sleep 60; done; lane 13 "belief:--mode belief --k 32" "slots:--mode slots --k 32";;
 C) while [ ! -f $O/base_s13/test_summary.json ]; do sleep 60; done; lane 13 "free:--mode free --k 32" "reread:--mode reread --k 32";;
 D) while [ ! -f $O/base_s13/test_summary.json ]; do sleep 60; done; lane 13 "ic:--mode indep" "quoteR2:--mode quote --k 32 --rounds 2";;
 E) lane 29 "ic:--mode indep" "quote:--mode quote --k 32" "belief:--mode belief --k 32";;
 F) lane 41 "ic:--mode indep" "quote:--mode quote --k 32" "belief:--mode belief --k 32";;
esac
