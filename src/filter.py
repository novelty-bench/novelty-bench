import asyncio
import json

from aiofiles import open as aio_open
from datasets import load_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

with open("/home/yimingz3/secrets/openai-api-key", "r") as file:
    client = AsyncOpenAI(api_key=file.read().strip())

instances = (
    load_dataset("theblackcat102/sharegpt-english", split="train")
    .map(
        lambda x: {"text": x["conversations"][0]["text"]},
        remove_columns=["conversations"],
    )
    .shuffle()
    .select(range(200))
)


class PromptClassification(BaseModel):
    reasoning: str
    label: int


async def classify_prompt(instance: dict) -> dict:
    """Classifies a single prompt and returns the result."""
    prompt = instance["text"]
    messages = [
        {
            "role": "system",
            "content": (
                "You are helping to select prompts for a benchmark that measures language models' ability "
                "to generate diverse, high-quality alternative answers. Your task is to determine whether a given "
                "prompt naturally allows for multiple distinct valid responses.\n\n"
                "For a prompt to qualify, it should have multiple possible correct or valid answers, with each answer being meaningfully different. "
                "Prompts that seem to ask for a single answer qualify as long as the answer can vary based on different perspectives or criteria. "
                "For a prompt to be an ideal candidate, each different answer should provide additional utility for the user.\n\n"
                "Note that only English prompts that make a single, unambiguous request are considered for this benchmark. Incoherent, nonsensical, or incomplete prompts do not qualify.\n"
                "Please determine the quality of the given prompt for the benchmark on a Likert scale of 1 (Poor) to 5 (Excellent), and briefly explain "
                "your reasoning.\n\n"
                "Examples:\n\n"
                "Prompt to evaluate: creating a banking system\n"
                "Reasoning: This prompt lacks specificity about what aspect of a banking system to create (e.g., database design, user interface, security features).\n"
                "Label: 1\n\n"
                "Prompt to evaluate: Let's play chess. I play whites. I play e4.\n"
                "Reasoning: The prompt requests a chess move in response to e4. Multiple valid responses exist (e.g., e5, c5, d5) that lead to distinct strategic positions and gameplay styles. Each response enables different types of games that could appeal to different players.\n"
                "Label: 5\n\n"
                "Prompt to evaluate: Write a Flask app that returns 'Hello, World!' when a user visits the root URL.\n"
                "Reasoning: While multiple implementations are possible (using different routing methods, error handling approaches, or project structures), these variations would be primarily syntactic rather than meaningful alternatives for most users seeking a basic Flask setup.\n"
                "Label: 3\n\n"
                "Prompt to evaluate: Übersetze meinen Text ins Französische\n"
                "Reasoning: This prompt does not qualify because it is not in English.\n"
                "Label: 1\n\n"
            ),
        },
        {
            "role": "user",
            "content": f"Prompt to evaluate: {prompt}\n\n",
        },
    ]

    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0,
            response_format=PromptClassification,
        )
        parsed_response = response.choices[0].message.parsed
        return {
            "id": instance["id"],
            "prompt": prompt,
            "label": parsed_response.label,
            "reasoning": parsed_response.reasoning,
        }
    except Exception as e:
        print(f"Error processing prompt '{prompt}': {e}")
        return {"prompt": prompt, "label": None, "reasoning": None}


async def process_prompts(instances, output_file):
    """Processes prompts concurrently and writes results to a file."""
    async with aio_open(output_file, "w") as f:
        tasks = []

        for instance in instances:
            tasks.append(classify_prompt(instance))

        for task in tqdm(asyncio.as_completed(tasks), total=len(instances)):
            result = await task
            await f.write(json.dumps(result) + "\n")


async def main():
    output_file = "data/filtered_sharegpt_sample.jsonl"
    await process_prompts(instances, output_file)
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
