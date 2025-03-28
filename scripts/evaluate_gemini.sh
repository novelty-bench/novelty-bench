#!/bin/bash
#SBATCH --job-name=sbb-gemini
#SBATCH --partition=preempt
#SBATCH --exclude=babel-3-21,babel-4-33,shire-1-10,babel-3-25,babel-1-23,babel-0-19
#SBATCH --requeue
#SBATCH --output=slurm_output/gemini_%j.out
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

concurrent=1
for model in "$@"; do
    # Set concurrent requests based on model name
    for data in curated wildchat
    do
        bash scripts/eval.sh $model $data eval/$data/gemini/$model --mode gemini --concurrent-requests $concurrent
    done
done

if [[ "$model" == "gemini-2.0-pro-exp-02-05" ]]; then
    bash scripts/eval.sh $model curated eval-ic/curated/gemini/$model --mode gemini --sampling in-context --concurrent-requests $concurrent
    bash scripts/eval.sh $model curated eval-paraphrase/curated/gemini/$model --mode gemini --sampling paraphrase --concurrent-requests $concurrent
    bash scripts/eval.sh $model curated eval-system-prompt/curated/gemini/$model --mode gemini --sampling system-prompt --concurrent-requests $concurrent
fi
