"""Independent-sample generation through the OpenAI or Anthropic batch API (half price).

submit  one batch per eval-dir: a request per (prompt, sample) for the prompts the
        cache says still need generating
status  request counts
collect writes generations.jsonl exactly as a live `inference.py --sampling
        regenerate` run would; missing or empty samples are regenerated live
"""

import argparse
import asyncio
import io
import json
import os
from collections import defaultdict

import anthropic
import openai
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from datasets import load_dataset

from src.evaluation_io import finalize, plan, stamp
from src.inference import (
    REFUSED,
    AnthropicService,
    OpenAIService,
    anthropic_params,
    anthropic_text,
    generation_config,
    openai_params,
    openai_text,
    validate_generations,
)

MANIFEST = "batch.generation.json"


def provider(model: str) -> str:
    return "anthropic" if model.startswith("claude") else "openai"


def gen_kwargs(config):
    return {
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "reasoning_effort": config["reasoning_effort"],
    }


def submit(eval_dir, prompts, config):
    output_file = os.path.join(eval_dir, "generations.jsonl")
    _, _, todo = plan(prompts, output_file, "generation_key", config)
    if not todo:
        return
    calls = [(x["id"], k) for x in todo for k in range(config["num_generations"])]
    bodies = {}
    for x in todo:
        messages = [{"role": "user", "content": x["prompt"]}]
        bodies[x["id"]] = (
            anthropic_params(config["model"], messages, **gen_kwargs(config))
            if config["mode"] == "anthropic"
            else openai_params(config["model"], messages, **gen_kwargs(config))
        )
    if config["mode"] == "anthropic":
        batch = anthropic.Anthropic().messages.batches.create(
            requests=[
                Request(
                    custom_id=f"c{n}",
                    params=MessageCreateParamsNonStreaming(**bodies[id_]),
                )
                for n, (id_, _) in enumerate(calls)
            ]
        )
    else:
        client = openai.OpenAI()
        lines = "".join(
            json.dumps(
                {
                    "custom_id": f"c{n}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": bodies[id_],
                }
            )
            + "\n"
            for n, (id_, _) in enumerate(calls)
        )
        upload = client.files.create(
            file=("batch.jsonl", io.BytesIO(lines.encode())), purpose="batch"
        )
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
    with open(os.path.join(eval_dir, MANIFEST), "w") as f:
        json.dump({"batch_id": batch.id, "config": config, "calls": calls}, f)
    print(f"{eval_dir}: submitted {batch.id} ({len(calls)} requests)")


def load_manifest(eval_dir):
    path = os.path.join(eval_dir, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def status(eval_dir):
    manifest = load_manifest(eval_dir)
    if not manifest:
        return
    if manifest["config"]["mode"] == "anthropic":
        b = anthropic.Anthropic().messages.batches.retrieve(manifest["batch_id"])
        print(f"{eval_dir}: {b.processing_status} {b.request_counts}")
    else:
        b = openai.OpenAI().batches.retrieve(manifest["batch_id"])
        print(f"{eval_dir}: {b.status} {b.request_counts}")


def fetch_results(manifest, raw_path):
    """Text per (prompt id, sample index) from a finished batch, or None if not done."""
    texts = defaultdict(dict)
    calls = manifest["calls"]
    if manifest["config"]["mode"] == "anthropic":
        client = anthropic.Anthropic()
        if (
            client.messages.batches.retrieve(manifest["batch_id"]).processing_status
            != "ended"
        ):
            return None
        with open(raw_path, "w") as f:
            for r in client.messages.batches.results(manifest["batch_id"]):
                f.write(json.dumps(r.model_dump(mode="json")) + "\n")
                id_, k = calls[int(r.custom_id[1:])]
                if r.result.type == "succeeded":
                    texts[id_][k] = anthropic_text(r.result.message)
                elif r.result.type == "errored" and "content filtering" in str(
                    r.result.error
                ):
                    texts[id_][k] = REFUSED
        return texts
    client = openai.OpenAI()
    b = client.batches.retrieve(manifest["batch_id"])
    if b.status != "completed":
        if b.status in {"failed", "expired", "cancelled"}:
            raise RuntimeError(f"batch {b.id} {b.status}: {b.errors}")
        return None
    content = client.files.content(b.output_file_id).text
    with open(raw_path, "w") as f:
        f.write(content)
    for line in content.splitlines():
        r = json.loads(line)
        id_, k = calls[int(r["custom_id"][1:])]
        if r.get("response") and r["response"]["status_code"] == 200:
            texts[id_][k] = openai_text(
                openai.types.chat.ChatCompletion.model_validate(r["response"]["body"])
            )
    return texts


async def collect(eval_dir, prompts):
    manifest = load_manifest(eval_dir)
    if not manifest:
        return
    config = manifest["config"]
    texts = fetch_results(
        manifest, os.path.join(eval_dir, "batch.generation.results.jsonl")
    )
    if texts is None:
        print(f"{eval_dir}: not finished")
        return

    output_file = os.path.join(eval_dir, "generations.jsonl")
    existing, keys, todo = plan(prompts, output_file, "generation_key", config)
    service = AnthropicService() if config["mode"] == "anthropic" else OpenAIService()
    n = config["num_generations"]
    counts = {"refused": 0, "regenerated": 0, "empty": 0}

    async def fill(x):
        got = texts.get(x["id"], {})
        gens = [got.get(k, "") for k in range(n)]
        for k in range(n):
            if gens[k].strip():
                continue
            counts["regenerated"] += 1
            for _ in range(2):
                (gens[k],) = await service.generate(
                    config["model"],
                    [{"role": "user", "content": x["prompt"]}],
                    **gen_kwargs(config),
                )
                if gens[k].strip():
                    break
            else:
                counts["empty"] += 1
                gens[k] = "[empty]"
        counts["refused"] += gens.count(REFUSED)
        validate_generations(gens, n)
        return stamp(
            x,
            {"model": config["model"], "generations": gens},
            "generation_key",
            keys[x["id"]],
            config,
        )

    with open(output_file + ".partial", "a") as journal:
        for row in await asyncio.gather(*(fill(x) for x in todo)):
            journal.write(json.dumps(row) + "\n")
            existing[row["id"]] = row
    finalize(prompts, output_file, existing)
    os.remove(os.path.join(eval_dir, MANIFEST))
    print(f"{eval_dir}: wrote {output_file} {counts}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["submit", "status", "collect"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="curated", choices=["curated", "wildchat"])
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    parser.add_argument(
        "--limit", type=int, help="only the first N prompts (smoke tests)"
    )
    args = parser.parse_args()

    config = generation_config(
        args.model,
        provider(args.model),
        "regenerate",
        args.num_generations,
        args.max_tokens,
        args.temperature,
        args.reasoning_effort,
    )
    os.makedirs(args.eval_dir, exist_ok=True)
    prompts = [
        dict(x) for x in load_dataset("yimingzhang/novelty-bench", split=args.data)
    ][: args.limit]
    match args.action:
        case "submit":
            submit(args.eval_dir, prompts, config)
        case "status":
            status(args.eval_dir)
        case "collect":
            await collect(args.eval_dir, prompts)


if __name__ == "__main__":
    asyncio.run(main())
