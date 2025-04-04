import argparse
import asyncio
import json
import os
import time
from abc import ABC, abstractmethod

import cohere
from aiofiles import open as aio_open
from anthropic import AsyncAnthropicVertex
from datasets import load_dataset
from google import genai
from google.auth import default, transport
from google.genai import types
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from src.common import oai_client


class InferenceService(ABC):
    @abstractmethod
    async def generate(
        self, model: str, messages: list[dict[str, str]], **kwargs
    ) -> list[str]: ...

    def cleanup(self):
        print("Done!")


class OpenAIService(InferenceService):
    def __init__(self):
        self.client = oai_client()

    async def generate(
        self, model: str, messages: list[dict[str, str]], **kwargs
    ) -> list[str]:
        resp = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return [c.message.content for c in resp.choices]


class TogetherService(OpenAIService):
    def __init__(self):
        with open("together-api-key") as file:
            self.client = AsyncOpenAI(
                api_key=file.read().strip(), base_url="https://api.together.xyz/v1"
            )


class VLLMService(OpenAIService):
    def __init__(self, model: str):
        port = int(os.environ["VLLM_PORT"])
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")


class CohereService(InferenceService):
    def __init__(self):
        with open("cohere-api-key") as file:
            self.client = cohere.AsyncClientV2(file.read().strip())

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, **kwargs
    ) -> list[str]:
        responses = []
        for _ in range(n):  # Cohere's API does not support parallel generation
            resp = await self.client.chat(model=model, messages=messages, **kwargs)
            responses.append(resp.message.content[0].text)
        return responses


class GeminiService(InferenceService):
    def __init__(self):
        with open("gemini-api-key") as file:
            self.client = genai.Client(api_key=file.read().strip())

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, max_tokens=512, **kwargs
    ) -> list[str]:
        contents = [
            types.Content(
                parts=[types.Part(text=msg["content"])],
                role="user" if msg["role"] == "user" else "model",
            )
            for msg in messages
        ]
        responses = []
        for _ in range(n):
            resp = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=max_tokens, **kwargs
                ),
            )
            if resp.candidates:
                responses.append(resp.candidates[0].content.parts[0].text)
            else:
                responses.append("[Blocked]")

        return responses


class AnthropicService(InferenceService):
    def __init__(self):
        self.client = AsyncAnthropicVertex(region="us-east5", project_id="GOOGLE-CLOUD-PROJECT-ID")

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, **kwargs
    ) -> list[str]:
        responses = []
        for _ in range(n):
            if messages[0]["role"] == "system":
                resp = await self.client.messages.create(
                    system=messages[0]["content"],
                    model=model,
                    messages=messages[1:],
                    **kwargs,
                )
                responses.append(resp.content[0].text)
            else:
                resp = await self.client.messages.create(
                    model=model, messages=messages, **kwargs
                )
                responses.append(resp.content[0].text)
        return responses


class VertexService(InferenceService):
    def __init__(self):
        self.client, self.last_refreshed = self.refresh_client()

    def refresh_client(self):
        model_location = "us-central1"
        project_id = "GOOGLE-CLOUD-PROJECT-ID"
        credentials, _ = default()
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)

        client = AsyncOpenAI(
            base_url=f"https://{model_location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{model_location}/endpoints/openapi/chat/completions?",
            api_key=credentials.token,
        )
        return client, time.time()

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, **kwargs
    ) -> list[str]:
        responses = []
        for _ in range(n):
            if time.time() - self.last_refreshed > 1800:
                self.client, self.last_refreshed = self.refresh_client()
            resp = await self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            responses.append(resp.choices[0].message.content)
        return responses


class DeepSeekService(OpenAIService):
    def __init__(self):
        with open("openrouter-api-key") as file:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=file.read().strip()
            )


async def run_generation(
    service: InferenceService,
    model: str,
    prompt: str,
    prompt_paraphrases: list[str] | None,
    num_generations: int,
    sampling: str,
    max_retries: int = 10,
) -> list[str]:
    responses = []
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            if sampling == "regenerate":
                # parallel generation w/o context
                responses = await service.generate(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    temperature=1.0,
                    n=num_generations,
                )

            elif sampling == "in-context":
                while len(responses) < num_generations:
                    response = await service.generate(
                        model=model,
                        messages=messages,
                        max_tokens=512,
                        temperature=1.0,
                    )
                    new_response = response[0]
                    responses.append(new_response)
                    messages.append({"role": "assistant", "content": new_response})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Can you generate a different answer?",
                        }
                    )

            elif sampling == "paraphrase":
                assert prompt_paraphrases and len(prompt_paraphrases) == num_generations
                while len(responses) < num_generations:
                    messages = [
                        {"role": "user", "content": prompt_paraphrases[len(responses)]}
                    ]
                    response = await service.generate(
                        model=model,
                        messages=messages,
                        max_tokens=512,
                        temperature=1.0,
                    )
                    new_response = response[0]
                    responses.append(new_response)

            elif sampling == "system-prompt":
                messages = [
                    {
                        "role": "system",
                        "content": "You are a producer of unique answers, and you strive to tell each user a novel answer to their question.",
                    },
                    {"role": "user", "content": prompt},
                ]
                responses = await service.generate(
                    model=model,
                    messages=messages,
                    max_tokens=512,
                    temperature=1.0,
                    n=num_generations,
                )
            else:
                raise Exception("Unknown mode " + sampling)

            return responses

        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(
                    f"Error generating response for prompt '{prompt}' after {max_retries} attempts: {e}",
                    flush=True,
                )
                return []

            # Exponential backoff
            wait_time = min(5 * 2**attempt, 60)  # 5, 10, 20, 40, 60, 60, ... seconds
            print(
                f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...",
                flush=True,
            )
            await asyncio.sleep(wait_time)


async def process_prompts(
    prompts,
    service,
    model,
    output_file,
    num_generations,
    concurrent_requests,
    sampling,
):
    """Processes all prompts concurrently and writes results to a file."""
    async with aio_open(output_file, "a", buffering=1) as f:
        semaphore = asyncio.Semaphore(concurrent_requests)

        async def process_single_prompt(prompt):
            async with semaphore:
                generations = await run_generation(
                    service,
                    model,
                    prompt["prompt"],
                    prompt.get("prompt_paraphrases"),
                    num_generations,
                    sampling,
                )
                return {
                    "id": prompt["id"],
                    "prompt": prompt["prompt"],
                    "model": model,
                    "generations": generations,
                }

        tasks = [process_single_prompt(prompt) for prompt in prompts]
        for task in tqdm(asyncio.as_completed(tasks), total=len(prompts)):
            result = await task
            await f.write(json.dumps(result) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "vllm",
            "openai",
            "together",
            "cohere",
            "gemini",
            "anthropic",
            "vertex",
            "deepseek",
        ],
        required=True,
        help="Inference service provider (vllm for local server, openai for API, etc.)",
    )
    parser.add_argument("--model", required=True, help="Model to run inference with")
    parser.add_argument(
        "--eval-dir", help="Directory to save evaluation results", required=True
    )
    parser.add_argument(
        "--data",
        default="curated",
        choices=["curated", "wildchat"],
        help="Source of prompts",
    )
    parser.add_argument(
        "--sampling",
        choices=["regenerate", "in-context", "paraphrase", "system-prompt"],
        default="regenerate",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=10,
        help="Number of generations per prompt",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    args = parser.parse_args()

    dataset = load_dataset("yimingzhang/novelty-bench", split=args.data)
    eval_dir = (
        args.eval_dir if args.eval_dir else os.path.join(f"{args.data}-evals", args.model)
    )
    os.makedirs(eval_dir, exist_ok=True)
    output_file = os.path.join(eval_dir, "generations.jsonl")

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        dataset_keys = set(dataset["id"])
        existing_output = load_dataset("json", data_files=output_file, split="train")
        existing_output = existing_output.filter(
            lambda x: len(x["generations"]) == args.num_generations
            and x["id"] in dataset_keys
        )

        # Save filtered dataset back to output file
        with open(output_file, "w") as f:
            for item in existing_output:
                f.write(json.dumps(item) + "\n")

        existing_keys = set(existing_output["id"])
        # Filter dataset to only include missing or invalid items
        dataset = dataset.filter(lambda x: x["id"] not in existing_keys)

        if len(dataset) == 0:
            print("All prompts have valid generations. Skipping.")
            return
        else:
            print(f"Generating {len(dataset)} missing or invalid entries.")

    concurrent_requests = args.concurrent_requests
    if args.mode == "vllm":
        service = VLLMService(args.model)
    elif args.mode == "openai":  # openai mode
        service = OpenAIService()
    elif args.mode == "together":
        service = TogetherService()
    elif args.mode == "cohere":
        service = CohereService()
    elif args.mode == "gemini":
        service = GeminiService()
    elif args.mode == "anthropic":
        service = AnthropicService()
    elif args.mode == "vertex":
        service = VertexService()
    elif args.mode == "deepseek":
        service = DeepSeekService()
    else:
        raise Exception(f"unknown service {service}")
    try:
        await process_prompts(
            dataset,
            service,
            args.model,
            output_file,
            args.num_generations,
            concurrent_requests,
            args.sampling,
        )

    finally:
        service.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
