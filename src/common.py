import functools
import os

import anthropic
from openai import AsyncOpenAI

# Each metric version keeps its own partitions/scores/summary under <eval-dir>/v<version>/.
METRIC_VERSIONS = ["1.0", "1.1"]
DEFAULT_VERSION = "1.1"
DEFAULT_JUDGE = "gpt-5.6-luna"


def version_dir(eval_dir: str, version: str) -> str:
    d = os.path.join(eval_dir, f"v{version}")
    os.makedirs(d, exist_ok=True)
    return d


def oai_client():
    """Prefer the local key file, otherwise use the SDK's environment credentials."""
    try:
        with open("openai-api-key") as file:
            return AsyncOpenAI(api_key=file.read().strip())
    except FileNotFoundError:
        return AsyncOpenAI()


def google_project(project=None, default_project=None):
    project = project or os.getenv("GOOGLE_CLOUD_PROJECT") or default_project
    if not project:
        raise ValueError("Set --project or GOOGLE_CLOUD_PROJECT for Vertex inference")
    return project


@functools.cache
def judge_client(model: str):
    client = anthropic.AsyncAnthropic() if model.startswith("claude") else oai_client()
    return client.with_options(max_retries=8)


def render_responses(prompt: str, responses: list[str]) -> str:
    parts = [f"<prompt>\n{prompt}\n</prompt>"]
    for i, r in enumerate(responses):
        parts.append(f'<response index="{i}">\n{r}\n</response>')
    return "\n\n".join(parts)


async def judge(model: str, system: str, user: str, schema):
    """One structured-output call; `schema` is a pydantic model."""
    client = judge_client(model)
    if model.startswith("claude"):
        msg = await client.messages.parse(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        return msg.parsed_output
    msg = await client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
    )
    return msg.choices[0].message.parsed
