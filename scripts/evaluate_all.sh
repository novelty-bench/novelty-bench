#!/bin/bash

set -Eeuo pipefail
# for source in anthropic
for source in cohere vllm gemini
# for source in vllm gemini cohere openai anthropic
do
    for model in $(cat model-lists/${source^^}_MODELS)
    do
        sbatch scripts/evaluate_$source.sh $model
    done
done
