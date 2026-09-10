import argparse
import asyncio
import bisect
import functools
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm.asyncio import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluation_io import cached_rows, evaluation_key, write_rows

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
        attn_implementation="eager",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return rm, tokenizer


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

    return score_first_occurrences(scores, partition)


def score_first_occurrences(scores, partition):
    assert len(scores) == len(partition)
    generation_scores, partition_scores = [], []
    seen = set()
    for score, label in zip(scores, partition, strict=True):
        generation_scores.append(0 if label in seen else score)
        if label not in seen:
            partition_scores.append(score)
            seen.add(label)
    return generation_scores, partition_scores


async def process_instances(instances, output_file, patience):
    if not 0 <= patience <= 1:
        raise ValueError("Patience must be between 0 and 1")
    existing = cached_rows(output_file)
    config = {"stage": "score", "version": 2, "patience": patience}
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async def process_single_instance(instance):
        key = evaluation_key(instance, config)
        cached = existing.get(instance["id"], {})
        if cached.get("score_key") == key:
            return cached
        async with semaphore:
            generation_scores, partition_scores = await score_partition_rm(
                instance["prompt"], instance["generations"], instance["partition"]
            )
        utility = np.average(
            generation_scores, weights=patience ** np.arange(len(generation_scores))
        )
        return {
            **instance,
            "generation_scores": generation_scores,
            "partition_scores": partition_scores,
            "utility": utility,
            "distinct": len(partition_scores),
            "score_key": key,
        }

    results = await tqdm.gather(*(process_single_instance(x) for x in instances))
    write_rows(output_file, results)


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
