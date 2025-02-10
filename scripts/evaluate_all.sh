#!/bin/bash

for source in vertex
# for source in openai anthropic cohere gemini vllm vertex
do
    for model in $(cat model-lists/${source^^}_MODELS)
    do
        sbatch scripts/evaluate_$source.sh $model
    done
done
