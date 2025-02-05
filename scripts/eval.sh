#!/bin/bash

set -Eeuo pipefail
set -x

MODEL=$1
EVAL_DIR=$2

shift 2
python src/inference.py --model $MODEL --eval-dir $EVAL_DIR $@
# python src/partition.py --model $MODEL --eval-dir $EVAL_DIR
# python src/score.py --model $MODEL --eval-dir $EVAL_DIR
# python src/summarize.py --model $MODEL --eval-dir $EVAL_DIR
