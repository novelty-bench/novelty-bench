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
import random

import numpy as np
from pydantic import BaseModel

from src.common import (
    DEFAULT_JUDGE,
    DEFAULT_VERSION,
    METRIC_VERSIONS,
    judge,
    render_responses,
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
A language model was asked the same prompt several times. Judge each response \
on its own.

valid: the response answers this prompt in the form and scope it asked for \
(format, length, count, language, other constraints), with no substantive error.

score, 1-10: 10 is the best response an expert could give to this prompt; 7 is \
good with clear room to improve; 4 is a weak but valid attempt; 1-3 are for \
invalid responses.

An unusual angle or format is fine if it answers the prompt. Extra length earns \
nothing."""


class Verdict(BaseModel):
    valid: bool
    score: int


class Scores(BaseModel):
    items: list[Verdict]


async def score_llm(prompt, generations, model, seed=None) -> list[int]:
    """Set-level judge; `seed` shuffles the order shown to the judge."""
    order = list(range(len(generations)))
    if seed is not None:
        random.Random(f"{seed}{prompt}").shuffle(order)
    shown = [generations[i] for i in order]
    for attempt in range(3):
        out = await judge(model, SCORE_SYSTEM, render_responses(prompt, shown), Scores)
        if len(out.items) == len(generations) and all(
            1 <= v.score <= 10 for v in out.items
        ):
            break
        print(f"invalid scores (attempt {attempt}): {out.items}")
    else:
        raise ValueError(f"judge never produced valid scores for {prompt!r}")
    scores = [0] * len(generations)
    for shown_i, orig in enumerate(order):
        v = out.items[shown_i]
        scores[orig] = v.score if v.valid else min(v.score, 3)
    return scores


SCORERS = {"rm": score_rm, "llm": score_llm}
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
        scores = await scorer(instance["prompt"], instance["generations"])
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
    parser.add_argument("--seed", type=int, help="shuffle response order shown to judge")
    parser.add_argument("--concurrency", type=int, default=1)
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
        scorer = functools.partial(scorer, model=args.judge_model, seed=args.seed)
        config |= {"judge_model": args.judge_model, "seed": args.seed}

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
