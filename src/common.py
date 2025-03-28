from openai import AsyncOpenAI

DATASETS = {
    "curated": "data/curated.jsonl",
    "wildchat": "data/wildchat-1k.jsonl",
    "debug": "data/debug.jsonl",
}


def oai_client():
    with open("openai-api-key") as file:
        return AsyncOpenAI(api_key=file.read().strip())
