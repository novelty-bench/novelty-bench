import argparse
import asyncio
import bisect
import functools
import json
import os

import datasets
import numpy as np
import torch
from aiofiles import open as aio_open
from datasets import load_dataset
from pydantic import BaseModel
from tqdm.asyncio import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CONCURRENT_REQUESTS = 1

reward_thresholds = [
    -7.71875,
    -6.28125,
    -6.0,
    -5.71875,
    -5.5,
    -5.0,
    -4.375,
    -3.4375,
    -2.046875,
]


def transform_raw_reward(reward: float) -> int:
    # score of 1 to 10
    return bisect.bisect_left(reward_thresholds, reward) + 1


@functools.cache
def rm_and_tokenizer():
    # Load model and tokenizer
    model_name = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return rm, tokenizer


class Rating(BaseModel):
    rating: int


@torch.inference_mode()
async def score_partition_rm(
    prompt: str, generations: list[str], partition: list[int]
) -> tuple[list[int], list[int]]:
    """Asynchronously scores the partition."""
    rm, tokenizer = rm_and_tokenizer()
    convs = [
        [
            {"content": prompt, "role": "user"},
            {"content": generation, "role": "assistant"},
        ]
        for generation in generations
    ]
    batch = tokenizer.apply_chat_template(
        convs,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(rm.device)
    # Get the reward scores
    with torch.no_grad():
        raw_rewards = rm(**batch).logits[:, 0].tolist()

    scores = [transform_raw_reward(r) for r in raw_rewards]

    generation_scores = []
    partition_scores = []

    for s, p in zip(scores, partition, strict=False):
        if p == len(partition_scores):
            generation_scores.append(s)
            partition_scores.append(s)
        else:
            generation_scores.append(0)

    assert len(partition_scores) == (max(partition) + 1), (
        f"partition_scores: {partition_scores}, partition: {partition}"
    )
    return generation_scores, partition_scores


async def process_instances(instances, output_file, patience):
    """Processes all instances concurrently and writes results to a file."""
    # Check if file exists and has matching keys
    if os.path.exists(output_file):
        try:
            existing_output = load_dataset("json", data_files=output_file, split="train")
            if not set(instances["id"]) - set(existing_output["id"]):
                print("All prompts are scored. Skipping.")
                return
        except datasets.exceptions.DatasetGenerationError:
            pass

    async with aio_open(output_file, "w", buffering=1) as f:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def process_single_instance(instance):
            async with semaphore:
                generation_scores, partition_scores = await score_partition_rm(
                    instance["prompt"],
                    instance["generations"],
                    instance["partition"],
                )
                utility = np.average(
                    generation_scores,
                    weights=patience ** np.arange(len(instance["generations"])),
                )
                return {
                    **instance,
                    "generation_scores": generation_scores,
                    "partition_scores": partition_scores,
                    "utility": utility,
                }

        tasks = [process_single_instance(instance) for instance in instances]

        for result in tqdm(await asyncio.gather(*tasks), total=len(instances)):
            await f.write(json.dumps(result) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    parser.add_argument(
        "--patience",
        help="Discount factor for computing cumulative utility.",
        type=float,
        default=0.8,
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    instances = load_dataset(
        "json",
        data_files=os.path.join(eval_dir, "partitions.jsonl"),
        split="train",
    )

    os.makedirs(eval_dir, exist_ok=True)

    output_file = os.path.join(eval_dir, "scores.jsonl")
    await process_instances(instances, output_file, args.patience)


if __name__ == "__main__":
    asyncio.run(main())
