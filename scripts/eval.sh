#!/bin/bash

set -Eeuo pipefail
set -x

MODEL=$1
DATA=$2
EVAL_DIR=$3

shift 3
# python src/inference.py --model $MODEL --data $DATA --eval-dir $EVAL_DIR $@
python src/partition.py --data $DATA --eval-dir $EVAL_DIR
python src/score.py --data $DATA --eval-dir $EVAL_DIR
python src/summarize.py --eval-dir $EVAL_DIR
