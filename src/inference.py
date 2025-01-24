import argparse
import asyncio
import json
import os

from aiofiles import open as aio_open
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.auto import tqdm

NUM_GENERATIONS = 10
CONCURRENT_REQUESTS = 50

with open("/home/yimingz3/secrets/openai-api-key", "r") as file:
    client = AsyncOpenAI(api_key=file.read().strip())


async def run_generation(model: str, prompt: str) -> list[str]:
    """Generates responses for a single prompt."""
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=1.0,
            n=NUM_GENERATIONS,
        )
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        print(f"Error generating response for prompt '{prompt}': {e}")
        return []


async def process_prompts(prompts, model, output_file):
    """Processes all prompts concurrently and writes results to a file."""
    async with aio_open(output_file, "w") as f:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process_single_prompt(prompt):
            async with semaphore:
                generations = await run_generation(model, prompt["prompt"])
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
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    args = parser.parse_args()

    dataset = load_dataset(
        "json", data_files="data/wildchat/dev-filtered.jsonl", split="train"
    )
    prompts = dataset.filter(lambda x: x["chosen"])

    os.makedirs(f"evals/{args.model}", exist_ok=True)

    output_file = f"evals/{args.model}/generations.jsonl"
    await process_prompts(prompts, args.model, output_file)

    print("done")


if __name__ == "__main__":
    asyncio.run(main())
