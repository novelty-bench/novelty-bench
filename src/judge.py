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


@dataclass(frozen=True)
class Stage:
    """One evaluation stage: the judge calls an instance needs, and how to fold the
    outputs into row fields. `fold` raises ValueError on invalid judge output."""

    name: str
    key_field: str
    calls: Callable[[dict, dict], list[Call]]
    fold: Callable[[dict, list, dict], dict]


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


async def run_stage(
    stage: Stage, instance: dict, config: dict, attempts: int = 3
) -> dict:
    """Judge one instance live: all its calls concurrently, then fold; retried on invalid output."""
    model, effort = config["judge_model"], config.get("effort", DEFAULT_EFFORT)
    for attempt in range(attempts):
        outputs = await asyncio.gather(
            *(judge(model, c, effort) for c in stage.calls(instance, config))
        )
        try:
            return stage.fold(instance, outputs, config)
        except ValueError as e:
            print(f"{stage.name}: invalid judge output (attempt {attempt}): {e}")
    # a persistently malformed answer is folded leniently rather than lost
    return stage.fold(instance, outputs, config | {"lenient": True})
