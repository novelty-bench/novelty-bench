#!/bin/bash

set -Eeuo pipefail
set -x

MODEL=$1

python src/inference.py --model $MODEL
python src/partition.py --model $MODEL
python src/score.py --model $MODEL
python src/summarize.py --model $MODEL

