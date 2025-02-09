#!/bin/bash
#SBATCH --job-name=sbb-vllm
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --output=slurm_output/vllm_%j.out
#SBATCH --gres=gpu:A100_80GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz3@cs.cmu.edu


export VLLM_CONFIGURE_LOGGING=0
export OMP_NUM_THREADS=12

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# babel-specific NCCL fix.
# TODO: remove for public release
if [[ "$(hostname)" =~ ^(babel-4-1|babel-4-5|babel-4-9|babel-4-13|babel-4-17|babel-4-21|babel-4-25|babel-4-29|babel-6-5|babel-6-9|babel-6-13|babel-7-1|babel-7-5|babel-7-9|babel-12-5|babel-12-9|babel-12-13|babel-13-1|babel-13-5|babel-13-9|babel-13-13|babel-13-17|babel-13-21|babel-13-25|babel-13-29|babel-14-1|babel-14-5|babel-14-9|babel-14-13|babel-14-17|babel-14-21|babel-14-25|babel-14-29|babel-14-37|babel-5-15|babel-5-19|babel-10-17|babel-6-29|babel-0-19|babel-11-25|shire-2-5|shire-2-9)$ ]]; then
  export NCCL_P2P_DISABLE=1
fi

concurrent=50
for model in "$@"; do
    # Get a random free port
    export VLLM_PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    
    # Start VLLM server in background
    bash scripts/vllm_server.sh "$model" $VLLM_PORT &
    SERVER_PID=$!
    
    # Wait for server to be healthy
    while ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null; do
        echo "Waiting for vLLM server to be ready..."
        sleep 5
    done
    echo "vLLM server is healthy"
    
    # Run all evaluations for this model
    for data in curated wildchat; do
        bash scripts/eval.sh "$model" "eval/$data/$model" --mode vllm --data "$data"
        bash scripts/eval.sh "$model" "eval-ic/$data/$model" --mode vllm --in-context --data "$data"
    done

    # Cleanup server for this model
    kill $SERVER_PID
    echo "Completed evaluations for $model"
done
