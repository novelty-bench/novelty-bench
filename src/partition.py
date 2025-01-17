import argparse
import asyncio
import json
import os
import random

from aiofiles import open as aio_open
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

CONCURRENT_REQUESTS = 50

with open("/home/yimingz3/secrets/openai-api-key", "r") as file:
    client = AsyncOpenAI(api_key=file.read().strip())


class Equivalence(BaseModel):
    equivalent: bool


async def equivalence_check(
    client: AsyncOpenAI, prompt: str, response_0: str, response_1: str
) -> bool:
    """Asynchronously checks equivalence between two responses."""
    messages = [
        {
            "role": "system",
            "content": "For a given prompt, determine whether the two responses are semantically equivalent.",
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    "Prompt: " + prompt,
                    "Response A: " + response_0,
                    "Response B: " + response_1,
                ]
            ),
        },
    ]

    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            max_tokens=10,
            temperature=0,
            response_format=Equivalence,
        )
        return response.choices[0].message.parsed.equivalent
    except Exception as e:
        print(f"Error in equivalence check: {e}")
        return False


async def partition_responses(
    client: AsyncOpenAI, prompt: str, responses: list[str]
) -> list[list[str]]:
    """Partitions responses into equivalence classes."""
    equivalence_classes = []
    assigned = [False] * len(responses)

    for i in range(len(responses)):
        if assigned[i]:
            continue

        current_class = [responses[i]]
        assigned[i] = True

        for j in range(i + 1, len(responses)):
            if not assigned[j] and await equivalence_check(
                client, prompt, random.choice(current_class), responses[j]
            ):
                current_class.append(responses[j])
                assigned[j] = True

        equivalence_classes.append(current_class)

    return sorted(equivalence_classes, key=len, reverse=True)


async def process_instances(instances, output_file):
    """Processes all instances concurrently and writes results to a file."""
    async with aio_open(output_file, "w") as f:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process_single_instance(instance):
            async with semaphore:
                partition = await partition_responses(
                    client, instance["prompt"], instance["generations"]
                )
                return {**instance, "partition": partition}

        tasks = [process_single_instance(instance) for instance in instances]

        for task in tqdm(asyncio.as_completed(tasks), total=len(instances)):
            result = await task
            await f.write(json.dumps(result) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    args = parser.parse_args()

    instances = load_dataset(
        "json", data_files=f"evals/{args.model}/generations.jsonl", split="train"
    )

    os.makedirs(f"evals/{args.model}", exist_ok=True)

    # Process instances and save results
    output_file = f"evals/{args.model}/partitions.jsonl"
    await process_instances(instances, output_file)

    print("done")


if __name__ == "__main__":
    asyncio.run(main())
