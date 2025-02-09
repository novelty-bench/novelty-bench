#!/bin/bash
#SBATCH --job-name=sbb-anthropic
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --output=slurm_output/anthropic_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

for model in "$@"; do
    concurrent=5
    for data in curated wildchat
    do
        bash scripts/eval.sh $model eval/$data/anthropic/$model --mode anthropic --data $data --concurrent-requests $concurrent
        bash scripts/eval.sh $model eval-ic/$data/anthropic/$model --mode anthropic --in-context --data $data --concurrent-requests $concurrent
    done
done
