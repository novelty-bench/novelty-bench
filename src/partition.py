import argparse
import asyncio
import functools
import os

import sacrebleu
import torch
from datasets import load_dataset
from evaluate import load
from pydantic import BaseModel
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import oai_client
from src.evaluation_io import cached_rows, evaluation_key, write_rows

CONCURRENT_REQUESTS = 1

rouge_scorer = rouge_scorer.RougeScorer(["rouge1"])


@functools.cache
def load_judge_client():
    return oai_client()


@functools.cache
def load_bertscorer():
    return load("bertscore")


@functools.cache
def load_deberta_tokenizer_and_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    rouge_eval = rouge_scorer.score(s1, s2)
    return rouge_eval["rouge1"].fmeasure


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

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    iids = torch.tensor(input_ids, device=DEVICE, dtype=torch.int64)
    tids = torch.tensor(token_type_ids, device=DEVICE, dtype=torch.int64)

    with torch.inference_mode():
        outputs = model(input_ids=iids.unsqueeze(0), token_type_ids=tids.unsqueeze(0))
    score = outputs["logits"].softmax(-1)[0, 1]
    return score.cpu().item()


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
    scores = await bertscore(prompt, response_0, response_1)
    return scores > 0.719


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


async def partition_responses(
    prompt: str,
    responses: list[str],
    equivalence_alg,
) -> list[int]:
    """Partitions responses into equivalence classes."""
    if not responses:
        raise ValueError("At least one response is required")
    equivalence_classes = []
    partition = [-1] * len(responses)

    for i in range(len(responses)):
        if partition[i] >= 0:
            continue

        current_class = [responses[i]]
        partition[i] = len(equivalence_classes)

        for j in range(i + 1, len(responses)):
            if partition[j] == -1 and await equivalence_alg(
                prompt,
                current_class[0],
                responses[j],
            ):
                current_class.append(responses[j])
                partition[j] = len(equivalence_classes)

        equivalence_classes.append(current_class)

    assert all(p >= 0 for p in partition)
    return partition


EQUIVALENCE_ALGS = {
    "gpt4": equivalence_check_gpt4,
    "unigram": equivalence_check_unigram,
    "bertscore": equivalence_check_bertscore,
    "classifier": equivalence_check_classifier,
}


async def process_instances(instances, output_file, equivalence_alg):
    """Reuse only matching inputs/configuration; preserve output on failures."""
    existing = cached_rows(output_file)
    config = {"stage": "partition", "version": 2, "algorithm": equivalence_alg.__name__}
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async def process_single_instance(instance):
        key = evaluation_key(instance, config)
        cached = existing.get(instance["id"], {})
        if cached.get("partition_key") == key:
            return {**instance, **cached}
        async with semaphore:
            partition = await partition_responses(
                instance["prompt"], instance["generations"], equivalence_alg
            )
            return {
                **instance,
                "partition": partition,
                "distinct": len(set(partition)),
                "partition_key": key,
                "partition_config": config,
            }

    tasks = [process_single_instance(instance) for instance in instances]
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await task)
    write_rows(output_file, results)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg",
        default="classifier",
        help="Equivalence testing method",
        choices=EQUIVALENCE_ALGS,
    )
    parser.add_argument(
        "--eval-dir", help="Directory to save evaluation results", required=True
    )
    args = parser.parse_args()
    equivalence_alg = EQUIVALENCE_ALGS[args.alg]

    eval_dir = args.eval_dir
    instances = load_dataset(
        "json",
        data_files=os.path.join(eval_dir, "generations.jsonl"),
        split="train",
    )

    # Process instances and save results
    output_file = os.path.join(eval_dir, "partitions.jsonl")
    await process_instances(instances, output_file, equivalence_alg)


if __name__ == "__main__":
    asyncio.run(main())
