import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import requests
from aiofiles import open as aio_open
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from common import DATASETS, oai_client


def get_free_port():
    """Finds a free port to use for the VLLM server."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def run_generation(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    num_generations: int,
    in_context: bool = False,
) -> list[str]:
    """Generates responses for a single prompt using vllm server or oai_client."""
    messages = [{"role": "user", "content": prompt}]
    responses = []

    try:
        if in_context:
            for _ in range(num_generations):
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    temperature=1.0,
                )
                new_response = response.choices[0].message.content
                responses.append(new_response)
                messages.append({"role": "assistant", "content": new_response})
                messages.append(
                    {"role": "user", "content": "Can you generate a different answer?"}
                )
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=1.0,
                n=num_generations,
            )
            responses = [choice.message.content for choice in response.choices]

        return responses

    except Exception as e:
        print(f"Error generating response for prompt '{prompt}': {e}")
        return []


async def process_prompts(
    prompts,
    client,
    model,
    output_file,
    num_generations,
    concurrent_requests,
    in_context,
):
    """Processes all prompts concurrently and writes results to a file."""
    async with aio_open(output_file, "w") as f:
        semaphore = asyncio.Semaphore(concurrent_requests)

        async def process_single_prompt(prompt):
            async with semaphore:
                generations = await run_generation(
                    client, model, prompt["prompt"], num_generations, in_context
                )
                return {
                    "id": prompt["id"],
                    "prompt": prompt["prompt"],
                    "model": model,
                    "generations": generations,
                }

        tasks = [process_single_prompt(prompt) for prompt in prompts]
        for task in tqdm(asyncio.as_completed(tasks), total=len(prompts)):
            result = await task
            await f.write(json.dumps(result) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--run-vllm", action="store_true", help="Run VLLM server locally"
    )
    parser.add_argument("--data", default="curated", choices=DATASETS)
    parser.add_argument(
        "--in-context", action="store_true", help="Generate responses in context"
    )
    parser.add_argument("--eval-dir", help="Directory to save evaluation results")
    parser.add_argument(
        "--num-generations",
        type=int,
        default=10,
        help="Number of generations per prompt",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=50,
        help="Number of concurrent requests",
    )
    args = parser.parse_args()

    # Determine base URL

    # Start VLLM server if requested
    if args.run_vllm:
        free_port = get_free_port()
        base_url = f"http://localhost:{free_port}/v1"
        vllm_process = subprocess.Popen(
            ["vllm", "serve", args.model, "--port", str(free_port)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        # Block until VLLM server is available
        while True:
            try:
                requests.get(base_url)
                print("VLLM server is available")
                break
            except requests.ConnectionError:
                pass
            time.sleep(1)  # Wait for 1 second before retrying
        client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
        concurrent_requests = 9999
    else:
        assert "/" not in args.model, "you should probably be using --run-vllm"
        client = oai_client()
        concurrent_requests = args.concurrent_requests
    try:
        dataset = load_dataset("json", data_files=DATASETS[args.data], split="train")

        # Set evaluation directory
        eval_dir = (
            args.eval_dir
            if args.eval_dir
            else os.path.join(f"{args.data}-evals", args.model)
        )
        os.makedirs(eval_dir, exist_ok=True)

        output_file = os.path.join(eval_dir, "generations.jsonl")
        await process_prompts(
            dataset,
            client,
            args.model,
            output_file,
            args.num_generations,
            concurrent_requests,
            args.in_context,
        )

    finally:
        # Stop VLLM server if it was started
        if args.run_vllm:
            vllm_process.terminate()
            vllm_process.wait()
            print("Stopped VLLM server")


if __name__ == "__main__":
    asyncio.run(main())
