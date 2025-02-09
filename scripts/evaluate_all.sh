#!/bin/bash

for source in vllm
# for source in openai anthropic gemini together vllm
do
    for model in $(cat model-lists/${source^^}_MODELS)
    do
        sbatch scripts/evaluate_$source.sh $model
    done
done
