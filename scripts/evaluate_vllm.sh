#!/bin/bash
#SBATCH --job-name=sbb-vllm
#SBATCH --partition=preempt
#SBATCH --exclude=babel-3-21,babel-4-33,shire-1-10,babel-3-25,babel-1-23,babel-0-19
#SBATCH --requeue
#SBATCH --output=slurm_output/vllm_%j.out
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu

set -Eeuo pipefail

export VLLM_CONFIGURE_LOGGING=0
export OMP_NUM_THREADS=12

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# babel-specific NCCL fix.
# TODO: remove for public release
if [[ "$(hostname)" =~ ^(shire-2-(9|5)|babel-8-5|babel-4-(1|5|9|13|17|21|25|29)|babel-6-(5|9|13|29)|babel-7-(1|5|9)|babel-12-(5|9|13)|babel-13-(1|5|9|13|17|21|25|29)|babel-14-(1|5|9|13|17|21|25|29|37)|babel-5-15|babel-10-17|babel-0-19|babel-11-25|babel-9-3)$ ]]; then
  export NCCL_P2P_DISABLE=1
fi

concurrent=50
for model in "$@"; do
    # Get a random free port
    export VLLM_PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    
    # Start VLLM server in background
    # bash scripts/vllm_server.sh "$model" $VLLM_PORT &
    # SERVER_PID=$!
    
    # Wait for server to be healthy
    # while ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null; do
    #     echo "Waiting for vLLM server to be ready..."
    #     sleep 5
    # done
    # echo "vLLM server is healthy"
    
    # Run all evaluations for this model
    for data in curated wildchat; do
        bash scripts/eval.sh $model $data eval/$data/$model --mode vllm
    done

    # Cleanup server for this model
    # kill $SERVER_PID
    echo "Completed evaluations for $model"
done
