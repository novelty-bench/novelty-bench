import asyncio
import os

import pandas as pd
import tiktoken
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm


def oai_client():
    with open("openai-api-key") as file:
        return AsyncOpenAI(api_key=file.read().strip())


openai_client = oai_client()
llama_guard_client = AsyncOpenAI(
    api_key="EMPTY", base_url=f"http://localhost:{os.environ['VLLM_PORT']}/v1"
)


SYS_PROMPT = """You are helping select prompts for a benchmark that measures language models' ability to generate diverse, high-quality alternative answers. For a prompt to qualify, it should:
1. Allow diverse responses: The prompt should enable multiple valid distinct responses. For example, a prompt that asks for a salmon recipe, a chess move in a given position, or a continuation of a story would allow diverse responses. In contrast, a prompt that asks for a specific fact, or a rewrite of input text would not.
2. Both the prompt, and the desired response should be in English.
3. Request a natural language response. Requests that ask for code, images or other output formats do not qualify.
4. Make a single clearly interpretable request. For example, "recommend a reliable espresso machine" is clear, while "espresso machine" is not.

Classify the following prompt based on these criteria, and format the provided prompt. Output using the provided JSON format."""


gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4o")


class PromptClassification(BaseModel):
    allows_diverse_responses: bool
    is_english: bool
    is_natural_language: bool
    is_clear: bool
    formatted: str

    def chosen(self):
        return (
            self.allows_diverse_responses
            and self.is_english
            and self.is_natural_language
            and self.is_clear
        )


async def classify_prompt(instance: dict) -> dict:
    """Classifies a single prompt and returns the result."""
    prompt = instance["prompt"]
    is_safe = (
        await llama_guard_client.chat.completions.create(
            model="meta-llama/Llama-Guard-3-8B",
            messages=[
                {"content": prompt, "role": "user"},
            ],
            temperature=0.0,
        )
    ).choices[0].message.content.strip() == "safe"

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {
            "role": "user",
            "content": f"Prompt:\n{prompt}",
        },
    ]

    try:
        response = await openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            max_tokens=512,
            temperature=0,
            response_format=PromptClassification,
        )
        parsed = response.choices[0].message.parsed
        assert parsed, "Failed to parse"
        return instance | {
            "chosen": parsed.chosen() and is_safe,
            "prompt": parsed.formatted,
            "original_prompt": prompt,
            "safety": is_safe,
            "meta": {"response": parsed.model_dump()},
        }
    except Exception as e:
        print(f"Error processing prompt '{prompt}': {e}")
        return instance | {
            "chosen": False,
            "prompt": prompt,
            "original_prompt": prompt,
            "safety": is_safe,
            "meta": {"error": str(e)},
        }


async def process_prompts(instances) -> list[dict]:
    """Processes prompts concurrently and returns a list of results."""
    semaphore = asyncio.Semaphore(50)
    tasks = []

    async def process_single_prompt(instance):
        async with semaphore:
            return await classify_prompt(instance)

    for instance in instances:
        tasks.append(process_single_prompt(instance))

    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(instances)):
        result = await task
        results.append(result)

    return results


async def main():
    instances = load_dataset("json", data_files="data/wildchat/5k.jsonl", split="train")
    results = await process_prompts(instances)
    data = pd.DataFrame(results)
    data = data.sort_values(by="id")
    data.to_json("data/wildchat/5k-filtered.jsonl", orient="records")

    chosen = data[data["chosen"]].sample(frac=1.0).head(1000)
    chosen.to_json("data/wildchat-1k.jsonl", lines=True, orient="records")


if __name__ == "__main__":
    asyncio.run(main())
