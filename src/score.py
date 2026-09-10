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

from src.common import DEFAULT_JUDGE, DEFAULT_VERSION, METRIC_VERSIONS, version_dir
from src.evaluation_io import run_cached
from src.judge import DEFAULT_EFFORT, Call, Stage, judge_call

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


def heads(partition: list[int]) -> list[int]:
    """Index of the first generation in each class."""
    return [i for i, c in enumerate(partition) if c not in partition[:i]]


def score_calls(instance: dict, config: dict) -> list[Call]:
    """One call per class head, each judged without seeing its siblings."""
    return [
        Call(
            SCORE_SYSTEM,
            f"<prompt>\n{instance['prompt']}\n</prompt>\n\n<response>\n{instance['generations'][i]}\n</response>",
            Score,
        )
        for i in heads(instance["partition"])
    ]


def spread(instance: dict, outputs: list) -> list[int | None]:
    """Head scores in generation order; duplicates are left unscored."""
    scores = [None] * len(instance["generations"])
    for i, o in zip(heads(instance["partition"]), outputs, strict=True):
        if not 1 <= o.score <= 10:
            raise ValueError(f"score out of range: {o.score}")
        scores[i] = o.score
    return scores


def score_fold(instance: dict, outputs: list, config: dict) -> dict:
    scores = spread(instance, outputs)
    return utility_fields(scores, instance["partition"], config["patience"])


def score_refused(instance: dict) -> dict:
    """The judge declined this instance: nothing is scored, and the summary
    excludes it rather than counting a refusal as a bad answer."""
    return {
        "generation_scores": [None] * len(instance["generations"]),
        "partition_scores": [],
        "utility": None,
        "distinct": len(set(instance["partition"])),
        "unscored": "judge_refusal",
    }


STAGE = Stage("score", "score_key", score_calls, score_fold, score_refused)


async def score_llm(instance, model, fallback=None) -> list[int | None]:
    pairs = await asyncio.gather(
        *(
            judge_call(model, c, DEFAULT_EFFORT, fallback)
            for c in score_calls(instance, {})
        )
    )
    return spread(instance, [output for output, _ in pairs])


SCORERS = {
    "rm": lambda instance: score_rm(instance["prompt"], instance["generations"]),
    "llm": score_llm,
}
DEFAULT_SCORER = {"1.0": "rm", "1.1": "llm"}


def score_first_occurrences(scores, partition):
    """The first generation of each class is credited; later duplicates credit 0."""
    credited = [0] * len(scores)
    for i in heads(partition):
        credited[i] = scores[i]
    return credited, [scores[i] for i in heads(partition)]


def utility_fields(scores, partition, patience):
    credited, partition_scores = score_first_occurrences(scores, partition)
    return {
        "generation_scores": scores,
        "partition_scores": partition_scores,
        "utility": np.average(credited, weights=patience ** np.arange(len(credited))),
        "distinct": len(partition_scores),
    }


async def process_instances(
    instances, output_file, scorer, config, patience, concurrency=1
):
    if not 0 <= patience <= 1:
        raise ValueError("Patience must be between 0 and 1")

    async def compute(instance):
        return utility_fields(await scorer(instance), instance["partition"], patience)

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
        "--fallback-judge",
        default="claude-sonnet-5",
        help="scores the responses the main judge declines; not part of the cache key",
    )
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
        scorer = functools.partial(
            scorer, model=args.judge_model, fallback=args.fallback_judge
        )
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
