import asyncio
import json

import tiktoken
from aiofiles import open as aio_open
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

with open("openai-api-key") as file:
    client = AsyncOpenAI(api_key=file.read().strip())

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
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {
            "role": "user",
            "content": f"Prompt:\n{prompt}",
        },
    ]

    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            max_tokens=512,
            temperature=0,
            response_format=PromptClassification,
        )
        parsed = response.choices[0].message.parsed
        assert parsed, "Failed to parse"
        return instance | {
            "chosen": parsed.chosen(),
            "prompt": parsed.formatted,
            "original_prompt": prompt,
            "meta": {"response": parsed.model_dump()},
        }
    except Exception as e:
        print(f"Error processing prompt '{prompt}': {e}")
        return instance | {
            "chosen": False,
            "prompt": prompt,
            "original_prompt": prompt,
            "meta": {"error": str(e)},
        }


async def process_prompts(instances, output_file) -> float:
    """Processes prompts concurrently and writes results to a file."""
    corr = 0
    async with aio_open(output_file, "w") as f:
        tasks = []

        for instance in instances:
            tasks.append(classify_prompt(instance))

        for task in tqdm(asyncio.as_completed(tasks), total=len(instances)):
            result = await task
            corr += 1 if result["chosen"] == result["human_label"] else 0
            await f.write(json.dumps(result) + "\n")

    return corr / len(instances)


async def main():
    instances = load_dataset(
        "json", data_files="data/wildchat/human-agreement.jsonl", split="train"
    )
    output_file = "data/wildchat/human-agreement-labeled.jsonl"
    acc = await process_prompts(instances, output_file)
    print(
        f"accuracy: {acc:.1%}",
    )


if __name__ == "__main__":
    asyncio.run(main())
