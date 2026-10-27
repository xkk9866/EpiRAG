#!/bin/bash
cd "$(dirname "$0")/../.."
R="bash experiments/relay/run_local.sh"
$R belief_s13 --mode belief --seed 13
$R intra_s13 --mode intra --seed 13
$R relay_s29 --mode relay --seed 29
$R relay_s41 --mode relay --seed 41
$R relay_m1 --mode relay --slots 1 --seed 13
$R relay_m8 --mode relay --slots 8 --seed 13
$R relay_r2 --mode relay --rounds 2 --seed 13
echo QDONE
