from openai import AsyncOpenAI

DATASETS = {
    "curated": "data/curated.jsonl",
    "wildchat": "data/wildchat/dev-filtered.jsonl",
}


def oai_client():
    with open("/home/yimingz3/secrets/openai-api-key") as file:
        return AsyncOpenAI(api_key=file.read().strip())
