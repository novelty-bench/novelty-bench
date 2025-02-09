#!/bin/bash
#SBATCH --job-name=sbb-together
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --output=slurm_output/together_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

for model in "$@"; do
    # Set concurrent requests based on model
    if [ "$model" = "deepseek-ai/DeepSeek-R1" ]; then
        concurrent=1
    else
        concurrent=5
    fi

    for data in curated wildchat
    do
        bash scripts/eval.sh $model eval/$data/$model --mode together --data $data --concurrent-requests $concurrent
        bash scripts/eval.sh $model eval-ic/$data/$model --mode together --in-context --data $data --concurrent-requests $concurrent
    done
done
