#!/bin/bash

set -Eeuo pipefail
for source in openai anthropic cohere gemini vllm
do
    for model in $(cat model-lists/${source^^}_MODELS)
    do
        sbatch scripts/evaluate_$source.sh $model
    done
done
    