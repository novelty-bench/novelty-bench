from openai import AsyncOpenAI

DATASETS = {
    "curated": "data/curated.jsonl",
    "wildchat": "data/wildchat-1k.jsonl",
}


def oai_client():
    with open("/home/yimingz3/secrets/openai-api-key") as file:
        return AsyncOpenAI(api_key=file.read().strip())


def together_client():
    with open("/home/yimingz3/secrets/together-api-key") as file:
        return AsyncOpenAI(
            api_key=file.read().strip(), base_url="https://api.together.xyz/v1"
        )
