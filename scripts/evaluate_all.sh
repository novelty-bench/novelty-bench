#!/bin/bash

for source in openai anthropic cohere gemini together vllm
do
    for model in $(cat model-lists/${source^^}_MODELS)
    do
        sbatch scripts/evaluate_$source.sh $model
    done
done
