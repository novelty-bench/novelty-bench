import argparse
import asyncio
import json
import os

from aiofiles import open as aio_open
from datasets import load_dataset
from tqdm.auto import tqdm

from common import DATASETS, oai_client

client = oai_client()


async def run_generation(
    model: str, prompt: str, num_generations: int, in_context: bool = False
) -> list[str]:
    """Generates responses for a single prompt."""
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
    prompts, model, output_file, num_generations, concurrent_requests, in_context
):
    """Processes all prompts concurrently and writes results to a file."""
    async with aio_open(output_file, "w") as f:
        semaphore = asyncio.Semaphore(concurrent_requests)

        async def process_single_prompt(prompt):
            async with semaphore:
                generations = await run_generation(
                    model, prompt["prompt"], num_generations, in_context
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
        args.model,
        output_file,
        args.num_generations,
        args.concurrent_requests,
        args.in_context,
    )

    print("done")


if __name__ == "__main__":
    asyncio.run(main())
