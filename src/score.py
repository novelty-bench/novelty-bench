import argparse
import asyncio
import bisect
import functools
import json
import os

import numpy as np
import torch
from aiofiles import open as aio_open
from datasets import load_dataset
from pydantic import BaseModel
from tqdm.asyncio import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import DATASETS

CONCURRENT_REQUESTS = 1

reward_percentiles = [
    -25.0,
    -15.0,
    -13.0,
    -12.0,
    -11.25,
    -10.625,
    -10.125,
    -9.625,
    -9.25,
    -8.875,
    -8.5,
    -8.1875,
    -7.90625,
    -7.65625,
    -7.40625,
    -7.15625,
    -6.96875,
    -6.78125,
    -6.625,
    -6.4375,
    -6.3125,
    -6.15625,
    -6.03125,
    -5.90625,
    -5.78125,
    -5.6875,
    -5.5625,
    -5.4375,
    -5.34375,
    -5.25,
    -5.15625,
    -5.0625,
    -4.96875,
    -4.90625,
    -4.8125,
    -4.71875,
    -4.65625,
    -4.5625,
    -4.5,
    -4.40625,
    -4.3125,
    -4.25,
    -4.15625,
    -4.09375,
    -4.03125,
    -3.953125,
    -3.875,
    -3.796875,
    -3.734375,
    -3.65625,
    -3.578125,
    -3.515625,
    -3.4375,
    -3.359375,
    -3.294999999999959,
    -3.21875,
    -3.140625,
    -3.0625,
    -3.0,
    -2.921875,
    -2.84375,
    -2.765625,
    -2.6875,
    -2.609375,
    -2.515625,
    -2.453125,
    -2.359375,
    -2.28125,
    -2.203125,
    -2.109375,
    -2.03125,
    -1.9375,
    -1.8515625,
    -1.75,
    -1.65625,
    -1.56640625,
    -1.46875,
    -1.3671875,
    -1.2578125,
    -1.1484375,
    -1.03125,
    -0.91015625,
    -0.77734375,
    -0.640625,
    -0.4921875,
    -0.341796875,
    -0.181640625,
    -0.0028903198242182936,
    0.1904296875,
    0.3984375,
    0.6374999999999886,
    0.91015625,
    1.2265625,
    1.6015625,
    2.0,
    2.5,
    3.15625,
    4.03125,
    5.34375,
    7.625,
    float("inf"),
]


def transform_raw_reward(reward: float) -> int:
    return bisect.bisect_left(reward_percentiles, reward)


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
    # if os.path.exists(output_file):
    #     try:
    #         existing_output = load_dataset("json", data_files=output_file, split="train")
    #         if not set(instances["id"]) - set(existing_output["id"]):
    #             print("All prompts are scored. Skipping.")
    #             return
    #     except datasets.exceptions.DatasetGenerationError:
    #         pass

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
    parser.add_argument("--data", default="curated", choices=DATASETS)
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
