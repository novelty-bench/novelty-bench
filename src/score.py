"""Score every generation, then derive per-class and cumulative utility from the partition.

v1.0 used a reward model and only kept scores for class heads; v1.1 scores all
generations with an LLM judge so partitions and scores can be revised independently.
"""

import argparse
import asyncio
import bisect
import functools
import json
import os

import numpy as np
from pydantic import BaseModel

from src.common import (
    DEFAULT_JUDGE,
    DEFAULT_VERSION,
    METRIC_VERSIONS,
    judge,
    version_dir,
)
from src.evaluation_io import run_cached

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
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


async def score_rm(prompt: str, generations: list[str]) -> list[int]:
    import torch

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
    with torch.inference_mode():
        raw_rewards = rm(**batch).logits[:, 0].tolist()
    return [transform_raw_reward(r) for r in raw_rewards]


SCORE_SYSTEM = """\
Score the response to the prompt, 1-10, by how well it serves what the prompt \
asked for.

A prompt that asks for one thing is best served by one thing; elaboration \
beyond the ask earns nothing, and craft counts where the prompt calls for it.
10: the best response an expert could give to this prompt.
7: good, with clear room to improve.
4: a weak attempt that still answers the prompt.
1-3: fails the ask: wrong form or scope, off-topic, empty, or a substantive error.

An unusual angle or format is fine if it answers the prompt."""


class Score(BaseModel):
    score: int


async def score_one(model: str, prompt: str, response: str) -> int:
    user = f"<prompt>\n{prompt}\n</prompt>\n\n<response>\n{response}\n</response>"
    for attempt in range(3):
        out = await judge(model, SCORE_SYSTEM, user, Score)
        if 1 <= out.score <= 10:
            return out.score
        print(f"invalid score (attempt {attempt}): {out.score}")
    raise ValueError(f"judge never produced a valid score for {prompt!r}")


async def score_llm(instance, model) -> list[int]:
    """Each generation is judged on its own, without seeing its siblings."""
    return await asyncio.gather(
        *(score_one(model, instance["prompt"], g) for g in instance["generations"])
    )


SCORERS = {
    "rm": lambda instance: score_rm(instance["prompt"], instance["generations"]),
    "llm": score_llm,
}
DEFAULT_SCORER = {"1.0": "rm", "1.1": "llm"}


def score_first_occurrences(scores, partition):
    """The first generation of each class scores; later duplicates score 0."""
    generation_scores, partition_scores = [], []
    seen = set()
    for score, label in zip(scores, partition, strict=True):
        generation_scores.append(0 if label in seen else score)
        if label not in seen:
            partition_scores.append(score)
            seen.add(label)
    return generation_scores, partition_scores


async def process_instances(
    instances, output_file, scorer, config, patience, concurrency=1
):
    if not 0 <= patience <= 1:
        raise ValueError("Patience must be between 0 and 1")

    async def compute(instance):
        scores = await scorer(instance)
        credited, partition_scores = score_first_occurrences(
            scores, instance["partition"]
        )
        return {
            "generation_scores": scores,
            "partition_scores": partition_scores,
            "utility": np.average(credited, weights=patience ** np.arange(len(credited))),
            "distinct": len(partition_scores),
        }

    await run_cached(
        instances,
        output_file,
        "score_key",
        config | {"patience": patience},
        compute,
        concurrency,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, nargs="+")
    parser.add_argument("--version", default=DEFAULT_VERSION, choices=METRIC_VERSIONS)
    parser.add_argument("--scorer", choices=SCORERS)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--concurrency", type=int, default=1, help="instances in flight")
    parser.add_argument(
        "--patience",
        help="Discount factor for computing cumulative utility.",
        type=float,
        default=0.8,
    )
    args = parser.parse_args()

    name = args.scorer or DEFAULT_SCORER[args.version]
    scorer = SCORERS[name]
    config = {"stage": "score", "version": args.version, "scorer": name}
    if name == "llm":
        scorer = functools.partial(scorer, model=args.judge_model)
        config |= {"judge_model": args.judge_model}

    for eval_dir in args.eval_dir:
        vdir = version_dir(eval_dir, args.version)
        with open(os.path.join(vdir, "partitions.jsonl")) as f:
            instances = [json.loads(line) for line in f]
        await process_instances(
            instances,
            os.path.join(vdir, "scores.jsonl"),
            scorer,
            config,
            args.patience,
            args.concurrency,
        )


if __name__ == "__main__":
    asyncio.run(main())
