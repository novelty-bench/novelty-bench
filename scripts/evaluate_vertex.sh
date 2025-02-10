#!/bin/bash
#SBATCH --job-name=sbb-vertex
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --output=slurm_output/vertex_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

concurrent=1
for model in "$@"; do
    # Set concurrent requests based on model name
    for data in curated wildchat
    do
        bash scripts/eval.sh $model eval/$data/vertex/$model --mode vertex --data $data --concurrent-requests $concurrent
        bash scripts/eval.sh $model eval-ic/$data/vertex/$model --mode vertex --in-context --data $data --concurrent-requests $concurrent
    done
done
