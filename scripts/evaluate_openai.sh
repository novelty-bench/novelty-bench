#!/bin/bash
#SBATCH --job-name=sbb-openai
#SBATCH --partition=general
#SBATCH --exclude=babel-3-21,babel-4-33,shire-1-10,babel-3-25
#SBATCH --requeue
#SBATCH --output=slurm_output/openai_%j.out
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
        bash scripts/eval.sh $model $data eval/$data/openai/$model --mode openai --concurrent-requests $concurrent
        bash scripts/eval.sh $model $data eval-ic/$data/openai/$model --mode openai --in-context --concurrent-requests $concurrent
    done
done
