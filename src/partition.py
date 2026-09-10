"""Partition each prompt's generations into equivalence classes.

v1.0 algorithms judge pairs (a response joins the first class whose head it
matches); the v1.1 `llm` algorithm sees all generations at once.
"""

import argparse
import asyncio
import functools
import json
import os

import sacrebleu
import torch
from pydantic import BaseModel
from rouge_score import rouge_scorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import (
    DEFAULT_JUDGE,
    DEFAULT_VERSION,
    METRIC_VERSIONS,
    oai_client,
    render_responses,
    version_dir,
)
from src.evaluation_io import run_cached
from src.judge import Call, Stage, run_stage, shuffled_order

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rouge_scorer = rouge_scorer.RougeScorer(["rouge1"])


@functools.cache
def load_judge_client():
    return oai_client()


@functools.cache
def load_bertscorer():
    from evaluate import load

    return load("bertscore")


@functools.cache
def load_deberta_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(
        "yimingzhang/deberta-v3-large-generation-similarity"
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


async def bleu(prompt: str, s1: str, s2: str):
    return (
        sacrebleu.corpus_bleu([s1], [[s2]]).score
        + sacrebleu.corpus_bleu([s2], [[s1]]).score
    ) / 200


async def rouge1(prompt: str, s1: str, s2: str):
    return rouge_scorer.score(s1, s2)["rouge1"].fmeasure


async def bertscore(prompt: str, s1: str, s2: str):
    return load_bertscorer().compute(
        predictions=[s1],
        references=[s2],
        model_type="microsoft/deberta-large",
    )["f1"][0]


async def classifier_score(prompt: str, s1: str, s2: str):
    tokenizer, model = load_deberta_tokenizer_and_model()
    input_ids = [tokenizer.cls_token_id]
    for s in [s1, s2]:
        input_ids.extend(
            tokenizer.encode(
                s,
                truncation=True,
                max_length=128,
                add_special_tokens=False,
            )
        )
        input_ids.append(tokenizer.sep_token_id)
        prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
    token_type_ids = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

    iids = torch.tensor(input_ids, device=DEVICE, dtype=torch.int64)
    tids = torch.tensor(token_type_ids, device=DEVICE, dtype=torch.int64)

    with torch.inference_mode():
        outputs = model(input_ids=iids.unsqueeze(0), token_type_ids=tids.unsqueeze(0))
    score = outputs["logits"].softmax(-1)[0, 1]
    return score.cpu().item()


async def equivalence_check_gpt4(prompt: str, response_0: str, response_1: str) -> bool:
    class Equivalence(BaseModel):
        equivalent: bool

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
                ],
            ),
        },
    ]

    response = await load_judge_client().beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        max_tokens=64,
        temperature=0,
        response_format=Equivalence,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Equivalence judge returned no parsed result")
    return parsed.equivalent


async def equivalence_check_unigram(
    prompt: str, response_0: str, response_1: str
) -> bool:
    return await rouge1(prompt, response_0, response_1) > 0.458


async def equivalence_check_bertscore(
    prompt: str,
    response_0: str,
    response_1: str,
) -> bool:
    return await bertscore(prompt, response_0, response_1) > 0.719


def maybe_test_equality(response_0: str, response_1: str) -> bool | None:
    # Only exact text equality is a safe shortcut: overlap can merge opposites
    # such as "do it" / "do not do it". Preserve case and whitespace semantics.
    if response_0 == response_1:
        return True
    return None


async def equivalence_check_classifier(
    prompt: str,
    response_0: str,
    response_1: str,
) -> bool:
    equality = maybe_test_equality(response_0, response_1)
    if equality is not None:
        return equality
    score = await classifier_score(prompt, response_0, response_1)
    return score > 0.102


async def partition_pairwise(prompt, responses, equivalence_alg) -> list[int]:
    """Each response joins the first class whose head it matches."""
    if not responses:
        raise ValueError("At least one response is required")
    heads = []
    partition = [-1] * len(responses)
    for i, r in enumerate(responses):
        for c, head in enumerate(heads):
            if await equivalence_alg(prompt, head, r):
                partition[i] = c
                break
        else:
            partition[i] = len(heads)
            heads.append(r)
    return partition


JUDGE_SYSTEM = """\
A language model was asked the same prompt several times. Partition its \
responses into groups of equivalent answers.

Two responses are equivalent if a user who had read one would gain essentially \
nothing from the other: the same central answer, recommendation, plot, argument \
or image, differing only in wording, formatting, length, ordering or minor \
detail. They are distinct if the central content differs, even when structure \
or register is shared. A list is distinct only if most of its items differ."""


class Partition(BaseModel):
    groups: list[list[int]]


def partition_calls(instance: dict, config: dict) -> list[Call]:
    prompt, responses = instance["prompt"], instance["generations"]
    order = shuffled_order(prompt, len(responses), config.get("seed"))
    shown = [responses[i] for i in order]
    return [Call(JUDGE_SYSTEM, render_responses(prompt, shown), Partition)]


def partition_fold(instance: dict, outputs: list, config: dict) -> dict:
    n = len(instance["generations"])
    order = shuffled_order(instance["prompt"], n, config.get("seed"))
    groups = outputs[0].groups
    if sorted(i for g in groups for i in g) != list(range(n)):
        raise ValueError(f"not a partition of {n} responses: {groups}")
    partition = [0] * n
    for g, members in enumerate(groups):
        for shown in members:
            partition[order[shown]] = g
    partition = canonical(partition)
    return {"partition": partition, "distinct": len(set(partition))}


STAGE = Stage("partition", "partition_key", partition_calls, partition_fold)


async def partition_llm(prompt, responses, model, seed=None) -> list[int]:
    """Set-level judge; `seed` shuffles the order shown to the judge."""
    instance = {"id": prompt[:40], "prompt": prompt, "generations": responses}
    fields = await run_stage(STAGE, instance, {"judge_model": model, "seed": seed})
    return fields["partition"]


def canonical(partition: list[int]) -> list[int]:
    """Relabel classes 0, 1, 2... in order of first appearance."""
    seen: dict[int, int] = {}
    return [seen.setdefault(g, len(seen)) for g in partition]


PARTITION_ALGS = {
    "gpt4": functools.partial(partition_pairwise, equivalence_alg=equivalence_check_gpt4),
    "unigram": functools.partial(
        partition_pairwise, equivalence_alg=equivalence_check_unigram
    ),
    "bertscore": functools.partial(
        partition_pairwise, equivalence_alg=equivalence_check_bertscore
    ),
    "classifier": functools.partial(
        partition_pairwise, equivalence_alg=equivalence_check_classifier
    ),
    "llm": partition_llm,
}
DEFAULT_ALG = {"1.0": "classifier", "1.1": "llm"}

INPUT_FIELDS = ["id", "prompt", "generations", "model", "prompt_paraphrases"]


def load_instances(eval_dir: str) -> list[dict]:
    """Read generations.jsonl, or recover generations from a prior version's output
    (versioned subdir, or the pre-1.1 flat layout)."""
    candidates = [os.path.join(eval_dir, "generations.jsonl")] + [
        os.path.join(eval_dir, sub, f)
        for sub in [f"v{v}" for v in METRIC_VERSIONS] + [""]
        for f in ["partitions.jsonl", "scores.jsonl"]
    ]
    path = next(p for p in candidates if os.path.exists(p))
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return [{k: r[k] for k in INPUT_FIELDS if k in r} for r in rows]


async def process_instances(instances, output_file, partition_alg, config, concurrency=1):
    async def compute(instance):
        partition = await partition_alg(instance["prompt"], instance["generations"])
        return {"partition": partition, "distinct": len(set(partition))}

    await run_cached(
        instances, output_file, "partition_key", config, compute, concurrency
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, nargs="+")
    parser.add_argument("--version", default=DEFAULT_VERSION, choices=METRIC_VERSIONS)
    parser.add_argument("--alg", choices=PARTITION_ALGS)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--seed", type=int, help="shuffle response order shown to judge")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    alg = args.alg or DEFAULT_ALG[args.version]
    partition_alg = PARTITION_ALGS[alg]
    config = {"stage": "partition", "version": args.version, "alg": alg}
    if alg == "llm":
        partition_alg = functools.partial(
            partition_alg, model=args.judge_model, seed=args.seed
        )
        config |= {"judge_model": args.judge_model, "seed": args.seed}

    for eval_dir in args.eval_dir:
        output_file = os.path.join(
            version_dir(eval_dir, args.version), "partitions.jsonl"
        )
        await process_instances(
            load_instances(eval_dir), output_file, partition_alg, config, args.concurrency
        )


if __name__ == "__main__":
    asyncio.run(main())
