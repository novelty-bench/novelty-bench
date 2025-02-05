#!/bin/bash
#SBATCH --job-name=sbb
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --output=slurm_output/logs_%j.out
#SBATCH --gres=gpu:6000Ada:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=6:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

# babel-specific NCCL fix.
# TODO: remove for public release
if [[ "$(hostname)" =~ ^(babel-4-1|babel-4-1|babel-4-1|babel-4-1|babel-4-5|babel-4-5|babel-4-5|babel-4-5|babel-4-9|babel-4-9|babel-4-9|babel-4-9|babel-4-13|babel-4-13|babel-4-13|babel-4-13|babel-4-17|babel-4-17|babel-4-17|babel-4-17|babel-4-21|babel-4-21|babel-4-21|babel-4-21|babel-4-25|babel-4-25|babel-4-25|babel-4-25|babel-4-29|babel-4-29|babel-4-29|babel-4-29|babel-6-5|babel-6-5|babel-6-5|babel-6-5|babel-6-9|babel-6-9|babel-6-9|babel-6-9|babel-6-13|babel-6-13|babel-6-13|babel-6-13|babel-7-1|babel-7-1|babel-7-1|babel-7-1|babel-7-5|babel-7-5|babel-7-5|babel-7-5|babel-7-9|babel-7-9|babel-7-9|babel-7-9|babel-12-5|babel-12-5|babel-12-5|babel-12-5|babel-12-9|babel-12-9|babel-12-9|babel-12-9|babel-12-13|babel-12-13|babel-12-13|babel-12-13|babel-13-1|babel-13-1|babel-13-1|babel-13-1|babel-13-5|babel-13-5|babel-13-5|babel-13-5|babel-13-9|babel-13-9|babel-13-9|babel-13-9|babel-13-13|babel-13-13|babel-13-13|babel-13-13|babel-13-17|babel-13-17|babel-13-17|babel-13-17|babel-13-21|babel-13-21|babel-13-21|babel-13-21|babel-13-25|babel-13-25|babel-13-25|babel-13-25|babel-13-29|babel-13-29|babel-13-29|babel-13-29|babel-14-1|babel-14-1|babel-14-1|babel-14-1|babel-14-5|babel-14-5|babel-14-5|babel-14-5|babel-14-9|babel-14-9|babel-14-9|babel-14-9|babel-14-13|babel-14-13|babel-14-13|babel-14-13|babel-14-17|babel-14-17|babel-14-17|babel-14-17|babel-14-21|babel-14-21|babel-14-21|babel-14-21|babel-14-25|babel-14-25|babel-14-25|babel-14-25|babel-14-29|babel-14-29|babel-14-29|babel-14-29|babel-14-37|babel-14-37|babel-5-15|babel-5-19|babel-10-17|babel-6-29|babel-0-19|babel-11-25|shire-2-5)$ ]]; then
  export NCCL_P2P_DISABLE=1
fi

while IFS= read -r model; do
    for data in wildchat curated
    do
        bash scripts/eval.sh $model eval/$data/$model --mode vllm --data $data
        bash scripts/eval.sh $model eval-ic/$data/$model --mode vllm --in-context --data $data
    done
done < "scripts/VLLM_MODELS"
