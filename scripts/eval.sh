#!/bin/bash

set -Eeuo pipefail
set -x

MODEL=$1

# python src/inference_async.py --model $MODEL
# python src/partition_async.py --model $MODEL
# python src/score_async.py --model $MODEL
python src/summarize.py --model $MODEL

