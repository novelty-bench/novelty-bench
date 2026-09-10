"""Run a judge stage through the Anthropic Message Batches API (half price, async).

submit  builds one batch per eval-dir from the instances the cache says need judging
status  reports request counts per batch
collect parses finished batches into the same rows a live run writes; instances
        whose batch output is missing or invalid are re-judged live
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from src import partition, score
from src.common import DEFAULT_JUDGE, DEFAULT_VERSION, METRIC_VERSIONS, version_dir
from src.evaluation_io import finalize, plan, stamp
from src.judge import (
    DEFAULT_EFFORT,
    JudgeRefusal,
    anthropic_params,
    parse_message,
    run_stage,
)

STAGES = {"partition": partition.STAGE, "score": score.STAGE}


def load(stage_name: str, eval_dir: str, vdir: str) -> list[dict]:
    if stage_name == "partition":
        return partition.load_instances(eval_dir)
    with open(os.path.join(vdir, "partitions.jsonl")) as f:
        return [json.loads(line) for line in f]


def manifest_path(vdir: str, stage_name: str) -> str:
    return os.path.join(vdir, f"batch.{stage_name}.json")


def submit(client, stage, eval_dir, vdir, config):
    instances = load(stage.name, eval_dir, vdir)
    output_file = os.path.join(vdir, f"{stage.name}s.jsonl")
    _, _, todo = plan(instances, output_file, stage.key_field, config)
    if not todo:
        return
    calls = [(x["id"], k, c) for x in todo for k, c in enumerate(stage.calls(x, config))]
    requests = [
        Request(
            custom_id=f"c{n}",  # instance ids may contain characters the API rejects
            params=MessageCreateParamsNonStreaming(
                **anthropic_params(config["judge_model"], call, config["effort"])
            ),
        )
        for n, (_, _, call) in enumerate(calls)
    ]
    batch = client.messages.batches.create(requests=requests)
    with open(manifest_path(vdir, stage.name), "w") as f:
        json.dump(
            {"batch_id": batch.id, "config": config, "calls": [c[:2] for c in calls]}, f
        )
    print(f"{vdir}: submitted {batch.id} ({len(requests)} requests)")


def status(client, vdir, stage_name):
    path = manifest_path(vdir, stage_name)
    if not os.path.exists(path):
        return
    with open(path) as f:
        batch = client.messages.batches.retrieve(json.load(f)["batch_id"])
    print(f"{vdir}: {batch.processing_status} {batch.request_counts}")


async def collect(client, stage, eval_dir, vdir):
    path = manifest_path(vdir, stage.name)
    if not os.path.exists(path):
        return
    with open(path) as f:
        manifest = json.load(f)
    config = manifest["config"]
    batch = client.messages.batches.retrieve(manifest["batch_id"])
    if batch.processing_status != "ended":
        print(f"{vdir}: {batch.processing_status}, skipping")
        return

    instances = load(stage.name, eval_dir, vdir)
    output_file = os.path.join(vdir, f"{stage.name}s.jsonl")
    existing, keys, todo = plan(instances, output_file, stage.key_field, config)

    raw = defaultdict(dict)
    with open(os.path.join(vdir, f"batch.{stage.name}.results.jsonl"), "w") as f:
        for r in client.messages.batches.results(manifest["batch_id"]):
            f.write(json.dumps(r.model_dump(mode="json")) + "\n")
            id_, k = manifest["calls"][int(r.custom_id[1:])]
            if r.result.type == "succeeded":
                raw[id_][k] = r.result.message

    retry = []
    with open(output_file + ".partial", "a") as journal:
        for x in todo:
            calls = stage.calls(x, config)
            try:
                outputs = [
                    parse_message(raw[x["id"]][k], c.schema) for k, c in enumerate(calls)
                ]
                fields = stage.fold(x, outputs, config)
            except JudgeRefusal:
                if stage.on_refusal is None:
                    raise
                print(f"{x['id']}: judge declined; recorded unscored")
                fields = stage.on_refusal(x)
            except (KeyError, ValueError) as e:
                print(f"{x['id']}: {e!r}; re-judging live")
                retry.append(x)
                continue
            row = stamp(x, fields, stage.key_field, keys[x["id"]], config)
            journal.write(json.dumps(row) + "\n")
            existing[x["id"]] = row

        for x, result in zip(
            retry,
            await asyncio.gather(
                *(run_stage(stage, x, config) for x in retry), return_exceptions=True
            ),
            strict=True,
        ):
            if isinstance(result, Exception):
                print(f"{x['id']}: live retry failed: {result!r}")
                continue
            row = stamp(x, result, stage.key_field, keys[x["id"]], config)
            journal.write(json.dumps(row) + "\n")
            existing[x["id"]] = row

    missing = [x["id"] for x in instances if x["id"] not in existing]
    if missing:
        raise RuntimeError(
            f"{vdir}: {len(missing)} instances still unjudged; rerun collect"
        )
    finalize(instances, output_file, existing)
    os.remove(path)
    print(
        f"{vdir}: wrote {output_file} ({len(todo) - len(retry)} from batch, {len(retry)} retried)"
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["submit", "status", "collect"])
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--eval-dir", required=True, nargs="+")
    parser.add_argument("--version", default=DEFAULT_VERSION, choices=METRIC_VERSIONS)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--effort", default=DEFAULT_EFFORT)
    parser.add_argument("--seed", type=int, help="partition: shuffle response order")
    parser.add_argument("--patience", type=float, default=0.8)
    args = parser.parse_args()

    stage = STAGES[args.stage]
    config = {
        "stage": stage.name,
        "version": args.version,
        "judge_model": args.judge_model,
        "effort": args.effort,
    }
    if stage.name == "partition":
        config |= {"alg": "llm", "seed": args.seed}
    else:
        config |= {"scorer": "llm", "patience": args.patience}

    client = anthropic.Anthropic()
    for eval_dir in args.eval_dir:
        vdir = version_dir(eval_dir, args.version)
        match args.action:
            case "submit":
                submit(client, stage, eval_dir, vdir, config)
            case "status":
                status(client, vdir, stage.name)
            case "collect":
                await collect(client, stage, eval_dir, vdir)


if __name__ == "__main__":
    asyncio.run(main())
