import argparse
import asyncio
import json
import os
import random

from aiofiles import open as aio_open
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

CONCURRENT_REQUESTS = 50


with open("/home/yimingz3/secrets/openai-api-key", "r") as file:
    client = AsyncOpenAI(api_key=file.read().strip())


class Rating(BaseModel):
    rating: int


async def score_partitions(prompt: str, partitions: list[list[str]]) -> list[int]:
    """Asynchronously scores the partitions."""
    ratings = []

    for partition in partitions:
        sample_response = random.choice(partition)
        messages = [
            {
                "role": "user",
                "content": f"[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 with the provided JSON format.\n\n[Question]\n{prompt}\n\n[The Start of Assistant's Answer]\n{sample_response}\n[The End of Assistant's Answer]",
            }
        ]

        try:
            resp = await client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=messages,
                max_tokens=100,
                temperature=0,
                response_format=Rating,
            )
            ratings.append(resp.choices[0].message.parsed.rating)
        except Exception as e:
            print(f"Error scoring partition: {e}")
            ratings.append(None)

    return ratings


async def process_instances(instances, output_file):
    """Processes all instances concurrently and writes results to a file."""
    async with aio_open(output_file, "w") as f:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process_single_instance(instance):
            async with semaphore:
                scores = await score_partitions(
                    instance["prompt"], instance["partition"]
                )
                return {**instance, "partition_scores": scores}

        tasks = [process_single_instance(instance) for instance in instances]

        for result in tqdm(await asyncio.gather(*tasks), total=len(instances)):
            await f.write(json.dumps(result) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    instances = load_dataset(
        "json", data_files=f"evals/{args.model}/partitions.jsonl", split="train"
    )

    os.makedirs(f"evals/{args.model}", exist_ok=True)

    output_file = f"evals/{args.model}/scores.jsonl"
    await process_instances(instances, output_file)

    print("done")


if __name__ == "__main__":
    asyncio.run(main())
