#!/bin/bash
#SBATCH --job-name=sbb-cohere
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --output=slurm_output/cohere_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

concurrent=5
for model in "$@"; do
    for data in curated wildchat
    do
        bash scripts/eval.sh $model eval/$data/cohere/$model --mode cohere --data $data --concurrent-requests $concurrent
        bash scripts/eval.sh $model eval-ic/$data/cohere/$model --mode cohere --in-context --data $data --concurrent-requests $concurrent
    done
done
