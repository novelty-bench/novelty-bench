import argparse
import asyncio
import functools
import json
import os
import random

from aiofiles import open as aio_open
from datasets import load_dataset
from evaluate import load
from openai import AsyncOpenAI
from pydantic import BaseModel
from rouge_score import rouge_scorer
from tqdm.auto import tqdm

CONCURRENT_REQUESTS = 50

with open("/home/yimingz3/secrets/openai-api-key", "r") as file:
    client = AsyncOpenAI(api_key=file.read().strip())

rouge_scorer = rouge_scorer.RougeScorer(["rouge1"])
bertscore = load("bertscore")


def rouge1(s1: str, s2: str):
    rouge_eval = rouge_scorer.score(s1, s2)
    return rouge_eval["rouge1"].fmeasure


async def equivalence_check_gpt4(prompt: str, response_0: str, response_1: str) -> bool:
    class Equivalence(BaseModel):
        equivalent: bool

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


async def equivalence_check_lcs(prompt: str, response_0: str, response_1: str) -> bool:
    return rouge1(response_0, response_1) > 0.5


async def equivalence_check_bertscore(
    prompt: str, response_0: str, response_1: str
) -> bool:
    scores = bertscore.compute(
        predictions=[response_0],
        references=[response_1],
        model_type="microsoft/deberta-large",
    )
    return scores["f1"][0] > 0.7


async def partition_responses(
    prompt: str, responses: list[str], equivalence_alg
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
            if not assigned[j] and await equivalence_alg(
                prompt, random.choice(current_class), responses[j]
            ):
                current_class.append(responses[j])
                assigned[j] = True

        equivalence_classes.append(current_class)

    return sorted(equivalence_classes, key=len, reverse=True)


EQUIVALENCE_ALGS = {
    "gpt4": equivalence_check_gpt4,
    "lcs": equivalence_check_lcs,
    "bertscore": equivalence_check_bertscore,
}


async def process_instances(instances, output_file, equivalence_alg):
    """Processes all instances concurrently and writes results to a file."""
    async with aio_open(output_file, "w") as f:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process_single_instance(instance):
            async with semaphore:
                partition = await partition_responses(
                    instance["prompt"], instance["generations"], equivalence_alg
                )
                return {**instance, "partition": partition}

        tasks = [process_single_instance(instance) for instance in instances]

        for task in tqdm(asyncio.as_completed(tasks), total=len(instances)):
            result = await task
            await f.write(json.dumps(result) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument(
        "--alg",
        default="bertscore",
        help="Equivalence testing method",
        choices=EQUIVALENCE_ALGS,
    )
    args = parser.parse_args()
    equivalence_alg = EQUIVALENCE_ALGS[args.alg]
    instances = load_dataset(
        "json", data_files=f"evals/{args.model}/generations.jsonl", split="train"
    )

    os.makedirs(f"evals/{args.model}", exist_ok=True)

    # Process instances and save results
    output_file = f"evals/{args.model}/partitions.jsonl"
    await process_instances(instances, output_file, equivalence_alg)

    print("done")


if __name__ == "__main__":
    asyncio.run(main())
