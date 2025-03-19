#!/bin/bash
#SBATCH --job-name=sbb-anthropic
#SBATCH --partition=preempt
#SBATCH --exclude=babel-3-21,babel-4-33,shire-1-10,babel-3-25,babel-1-23,babel-0-19,babel-1-23
#SBATCH --gres=gpu:A6000:2
#SBATCH --requeue
#SBATCH --output=slurm_output/anthropic_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

for model in "$@"; do
    concurrent=1
    for data in curated wildchat
    do
        bash scripts/eval.sh $model $data eval/$data/anthropic/$model --mode anthropic --concurrent-requests $concurrent
    done

    bash scripts/eval.sh $model curated eval-ic/curated/anthropic/$model --mode anthropic --sampling in-context --concurrent-requests $concurrent
    bash scripts/eval.sh $model curated eval-paraphrase/curated/anthropic/$model --mode anthropic --sampling paraphrase --concurrent-requests $concurrent
    bash scripts/eval.sh $model curated eval-system-prompt/curated/anthropic/$model --mode anthropic --sampling system-prompt --concurrent-requests $concurrent
done
