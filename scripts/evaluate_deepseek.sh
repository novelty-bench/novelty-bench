#!/bin/bash
#SBATCH --job-name=sbb-deepseek
#SBATCH --partition=preempt
#SBATCH --exclude=babel-3-21,babel-4-33,shire-1-10,babel-3-25,babel-1-23,babel-0-19
#SBATCH --requeue
#SBATCH --output=slurm_output/deepseek_%j.out
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

concurrent=50
for model in "$@"; do
    for data in curated wildchat
    do
        bash scripts/eval.sh $model $data eval/$data/$model --mode deepseek --concurrent-requests $concurrent
    done
done

if [[ "$model" == "deepseek/deepseek-r1" ]]; then
    bash scripts/eval.sh $model curated eval-ic/curated/$model --mode deepseek --sampling in-context --concurrent-requests $concurrent
    bash scripts/eval.sh $model curated eval-paraphrase/curated/$model --mode deepseek --sampling paraphrase --concurrent-requests $concurrent
    bash scripts/eval.sh $model curated eval-system-prompt/curated/$model --mode deepseek --sampling system-prompt --concurrent-requests $concurrent
fi
