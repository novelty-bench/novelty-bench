#!/bin/bash
#SBATCH --job-name=sbb
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --output=slurm_output/logs_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

for model in "$@"; do
    for data in wildchat curated
    do
        bash scripts/eval.sh $model eval/$data/$model --mode together --data $data --concurrent-requests 10
        bash scripts/eval.sh $model eval-ic/$data/$model --mode together --in-context --data $data --concurrent-requests 10
    done
done
