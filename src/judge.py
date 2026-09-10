"""Structured-output judge calls, shared by the live path and the Batches API."""

import asyncio
import functools
import random
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel

from src.common import oai_client

DEFAULT_EFFORT = "high"


@dataclass(frozen=True)
class Call:
    system: str
    user: str
    schema: type[BaseModel]


class JudgeRefusal(ValueError):
    """The judge declined to answer; no retry will change that."""


@dataclass(frozen=True)
class Stage:
    """One evaluation stage: the judge calls an instance needs, and how to fold the
    outputs into row fields. `fold` raises ValueError on invalid judge output;
    `on_refusal`, where a stage defines one, gives the fields to record instead
    when the judge declines the instance outright."""

    name: str
    key_field: str
    calls: Callable[[dict, dict], list[Call]]
    fold: Callable[[dict, list, dict], dict]
    on_refusal: Callable[[dict], dict] | None = None


def shuffled_order(prompt: str, n: int, seed: int | None) -> list[int]:
    order = list(range(n))
    if seed is not None:
        random.Random(f"{seed}{prompt}").shuffle(order)
    return order


def strict_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()

    def close(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for v in node.values():
                close(v)
        elif isinstance(node, list):
            for v in node:
                close(v)

    close(schema)
    return schema


def anthropic_params(model: str, call: Call, effort: str = DEFAULT_EFFORT) -> dict:
    return {
        "model": model,
        "max_tokens": 16000,
        "system": call.system,
        "messages": [{"role": "user", "content": call.user}],
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": strict_schema(call.schema)},
        },
    }


def parse_message(msg, schema: type[BaseModel]):
    if msg.stop_reason == "refusal":
        raise JudgeRefusal("judge declined to answer")
    if msg.stop_reason != "end_turn":
        raise ValueError(f"judge stopped with {msg.stop_reason}")
    text = next((b.text for b in msg.content if b.type == "text"), "")
    return schema.model_validate_json(text)


@functools.cache
def judge_client(model: str):
    client = anthropic.AsyncAnthropic() if model.startswith("claude") else oai_client()
    return client.with_options(max_retries=8)


async def judge(model: str, call: Call, effort: str = DEFAULT_EFFORT):
    client = judge_client(model)
    if model.startswith("claude"):
        msg = await client.messages.create(**anthropic_params(model, call, effort))
        return parse_message(msg, call.schema)
    msg = await client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": call.system},
            {"role": "user", "content": call.user},
        ],
        response_format=call.schema,
    )
    parsed = msg.choices[0].message.parsed
    if parsed is None:
        raise ValueError(
            f"judge returned no parsed output ({msg.choices[0].finish_reason})"
        )
    return parsed


async def judge_call(model: str, call: Call, effort: str, fallback: str | None = None):
    """One call, with a second judge for the answers this one declines.

    A refusal is deterministic and specific to the response being judged (a
    safety classifier firing on its content), so retrying the same model is
    pointless; another model usually answers. Returns (output, model used).
    """
    try:
        return await judge(model, call, effort), model
    except JudgeRefusal:
        if not fallback:
            raise
        return await judge(fallback, call, effort), fallback


async def run_stage(
    stage: Stage,
    instance: dict,
    config: dict,
    attempts: int = 3,
    fallback: str | None = None,
) -> dict:
    """Judge one instance live: all its calls concurrently, then fold; retried on invalid output.

    `fallback` is error recovery, not a metric setting, so it stays out of
    `config` and therefore out of the cache key; a row it touched says so.
    """
    model, effort = config["judge_model"], config.get("effort", DEFAULT_EFFORT)
    for attempt in range(attempts):
        calls = stage.calls(instance, config)
        try:
            pairs = await asyncio.gather(
                *(judge_call(model, c, effort, fallback) for c in calls)
            )
        except JudgeRefusal:
            if stage.on_refusal is None:
                raise
            print(f"{stage.name}: every judge declined {instance['id']}; unscored")
            return stage.on_refusal(instance)
        outputs = [output for output, _ in pairs]
        declined = [i for i, (_, used) in enumerate(pairs) if used != model]
        try:
            fields = stage.fold(instance, outputs, config)
        except ValueError as e:
            print(f"{stage.name}: invalid judge output (attempt {attempt}): {e}")
            continue
        if declined:
            print(f"{stage.name}: {instance['id']} calls {declined} scored by {fallback}")
            fields |= {"fallback_judge": fallback, "fallback_calls": declined}
        return fields
    # a persistently malformed answer is folded leniently rather than lost
    return stage.fold(instance, outputs, config | {"lenient": True})
