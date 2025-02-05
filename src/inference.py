import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import requests
import torch
from aiofiles import open as aio_open
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from common import DATASETS, oai_client, together_client


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
    in_context: bool,
    max_retries: int = 5,
) -> list[str]:
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
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
                        {
                            "role": "user",
                            "content": "Can you generate a different answer?",
                        }
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
            if attempt == max_retries - 1:  # Last attempt
                print(
                    f"Error generating response for prompt '{prompt}' after {max_retries} attempts: {e}"
                )
                return []

            # Exponential backoff
            wait_time = 2**attempt  # 1, 2, 4, 8, 16 seconds
            print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)


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
        "--mode",
        choices=["vllm", "openai", "together"],
        default="openai",
        help="Mode to run inference (vllm for local server, openai for API)",
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

    dataset = load_dataset("json", data_files=DATASETS[args.data], split="train")
    eval_dir = (
        args.eval_dir
        if args.eval_dir
        else os.path.join(f"{args.data}-evals", args.model)
    )
    os.makedirs(eval_dir, exist_ok=True)
    output_file = os.path.join(eval_dir, "generations.jsonl")

    if os.path.exists(output_file):
        existing_data = load_dataset("json", data_files=output_file, split="train")
        # check output file has matching ids
        if set(existing_data["id"]) == set(dataset["id"]):
            print("Output file already exists with matching IDs. Skipping generation.")
            return
        else:
            print("Output file exists but has different IDs. Overwriting.")

    concurrent_requests = args.concurrent_requests
    if args.mode == "vllm":
        free_port = get_free_port()
        base_url = f"http://localhost:{free_port}/v1"
        vllm_process = subprocess.Popen(
            [
                "vllm",
                "serve",
                args.model,
                "--port",
                str(free_port),
                "--tensor-parallel-size",
                str(torch.cuda.device_count()),
                "--max-model-len",
                "8192",
            ],
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
    elif args.mode == "openai":  # openai mode
        assert "/" not in args.model, "Local model paths should use --mode vllm"
        client = oai_client()
    elif args.mode == "together":
        client = together_client()
    try:
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
        if args.mode == "vllm":
            vllm_process.terminate()
            vllm_process.wait()
            print("Stopped VLLM server")


if __name__ == "__main__":
    asyncio.run(main())
