#!/bin/bash
#SBATCH --job-name=sbb-vertex
#SBATCH --partition=general
#SBATCH --exclude=babel-3-21,babel-4-33,shire-1-10,babel-3-25
#SBATCH --requeue
#SBATCH --output=slurm_output/vertex_%j.out
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

concurrent=5
for model in "$@"; do
    # Set concurrent requests based on model name
    for data in curated wildchat
    do
        bash scripts/eval.sh $model $data eval/$data/vertex/$model --mode vertex --concurrent-requests $concurrent
        bash scripts/eval.sh $model $data eval-ic/$data/vertex/$model --mode vertex --in-context --concurrent-requests $concurrent
    done
done
