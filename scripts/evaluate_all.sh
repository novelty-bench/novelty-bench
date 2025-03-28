#!/bin/bash

set -Eeuo pipefail
for source in deepseek #gemini
# for source in vllm gemini cohere openai anthropic deepseek
do
    for model in $(cat model-lists/${source^^}_MODELS)
    do
        sbatch scripts/evaluate_$source.sh $model
    done
done
