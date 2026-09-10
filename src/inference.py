import argparse
import asyncio
import json
import os
import time
from abc import ABC, abstractmethod

import cohere
import torch
from anthropic import AsyncAnthropic, AsyncAnthropicVertex, BadRequestError
from datasets import load_dataset
from google import genai
from google.auth import default, transport
from google.genai import types
from openai import AsyncOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import google_project, oai_client
from src.evaluation_io import run_cached


class InferenceService(ABC):
    @abstractmethod
    async def generate(
        self, model: str, messages: list[dict[str, str]], **kwargs
    ) -> list[str]: ...

    def cleanup(self):
        print("Done!")


REFUSED = "[refused]"  # provider-side refusal; placeholders keep the row valid
EMPTY = "[empty]"  # the model spent its whole token budget without answering


def log_usage(model: str, usage) -> None:
    """Append one line of token usage to $NB_USAGE_LOG, if set (no extra API calls)."""
    path = os.environ.get("NB_USAGE_LOG")
    if path and usage is not None:
        with open(path, "a") as f:
            f.write(
                json.dumps({"model": model, **usage.model_dump(exclude_none=True)}) + "\n"
            )


def openai_params(
    model, messages, max_tokens=512, temperature=1.0, reasoning_effort=None
):
    """Chat-completions body; reasoning models take an effort and no temperature."""
    body = {"model": model, "messages": messages}
    if reasoning_effort:  # the budget is shared with reasoning, so double it
        body |= {
            "max_completion_tokens": 2 * max_tokens,
            "reasoning_effort": reasoning_effort,
        }
    else:
        body |= {"max_tokens": max_tokens, "temperature": temperature}
    return body


def openai_text(completion) -> str:
    choice = completion.choices[0]
    if choice.finish_reason == "content_filter" or choice.message.refusal:
        return REFUSED
    text = choice.message.content or ""
    if not text.strip() and choice.finish_reason == "length":
        return EMPTY
    return text


def anthropic_params(
    model, messages, max_tokens=512, temperature=1.0, reasoning_effort=None
):
    """Messages body. With an effort: adaptive thinking, no sampling params, and a
    max_tokens budget doubled because thinking shares it with the visible answer.
    The last user turn carries a cache breakpoint so in-context runs reuse the prefix."""
    system = [m["content"] for m in messages if m["role"] == "system"]
    turns = [dict(m) for m in messages if m["role"] != "system"]
    turns[-1]["content"] = [
        {
            "type": "text",
            "text": turns[-1]["content"],
            "cache_control": {"type": "ephemeral"},
        }
    ]
    body = {"model": model, "messages": turns, "max_tokens": max_tokens}
    if system:
        body["system"] = "\n\n".join(system)
    if reasoning_effort:
        body |= {
            "max_tokens": 2 * max_tokens,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": reasoning_effort},
        }
    else:
        body["temperature"] = temperature
    return body


def anthropic_text(message) -> str:
    if message.stop_reason == "refusal":
        return REFUSED
    text = "".join(block.text for block in message.content if block.type == "text")
    if not text.strip() and message.stop_reason == "max_tokens":
        return EMPTY
    return text


class OpenAIService(InferenceService):
    def __init__(self):
        self.client = oai_client()

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, **kwargs
    ) -> list[str]:
        body = openai_params(model, messages, **kwargs)
        if "reasoning_effort" in body:  # reasoning models: one sample per request
            resps = await asyncio.gather(
                *(self.client.chat.completions.create(**body) for _ in range(n))
            )
            for r in resps:
                log_usage(model, r.usage)
            return [openai_text(r) for r in resps]
        resp = await self.client.chat.completions.create(n=n, **body)
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
        system = "\n\n".join(
            msg["content"] for msg in messages if msg["role"] == "system"
        )
        contents = [
            types.Content(
                parts=[types.Part(text=msg["content"])],
                role="user" if msg["role"] == "user" else "model",
            )
            for msg in messages
            if msg["role"] != "system"
        ]
        responses = []
        for _ in range(n):
            resp = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    system_instruction=system or None,
                    **kwargs,
                ),
            )
            if not resp.candidates or not resp.candidates[0].content:
                raise ValueError("Gemini returned no answer content")
            parts = resp.candidates[0].content.parts or []
            text = "".join(part.text for part in parts if part.text and not part.thought)
            if not text.strip():
                raise ValueError("Gemini returned no answer text")
            responses.append(text)
        return responses


class AnthropicService(InferenceService):
    """Anthropic API (ANTHROPIC_API_KEY)."""

    def __init__(self):
        self.client = AsyncAnthropic()

    async def generate(
        self, model: str, messages: list[dict[str, str]], n=1, **kwargs
    ) -> list[str]:
        body = anthropic_params(model, messages, **kwargs)
        return list(await asyncio.gather(*(self.create(body) for _ in range(n))))

    async def create(self, body) -> str:
        try:
            msg = await self.client.messages.create(**body)
            log_usage(body["model"], msg.usage)
            return anthropic_text(msg)
        except BadRequestError as e:  # output-side content filter is a 400, not a refusal
            if "content filtering" in str(e):
                return REFUSED
            raise


class AnthropicVertexService(AnthropicService):
    def __init__(self, project=None, region="us-east5"):
        self.client = AsyncAnthropicVertex(
            region=region, project_id=google_project(project)
        )


class VertexService(InferenceService):
    def __init__(self, project=None, region="us-central1"):
        self.project = project
        self.region = region
        self.client, self.last_refreshed = self.refresh_client()

    def refresh_client(self):
        model_location = self.region
        credentials, default_project = default()
        project_id = google_project(self.project, default_project)
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)

        client = AsyncOpenAI(
            base_url=f"https://{model_location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{model_location}/endpoints/openapi/",
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


class TransformersService(InferenceService):
    def __init__(self, model: str):
        self.model_name = model
        print(f"Loading tokenizer and model for {model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",  # Use flash attention if available
            )
        except (ImportError, ValueError):
            print("Flash attention not available, falling back to eager attention")
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
            )

        # Set pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on device: {self.model.device}")
        print(f"Model memory footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")
        print("Model loaded successfully!")

    async def generate(
        self,
        model: str,
        messages: list[dict[str, str]],
        n=1,
        max_tokens=512,
        temperature=1.0,
        stop=None,
        **kwargs,
    ) -> list[str]:
        # Run the actual generation in a thread to avoid blocking
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._generate_sync, messages, n, max_tokens, temperature, stop, kwargs
        )

    def _generate_sync(self, messages, n, max_tokens, temperature, stop, kwargs):
        if n < 1:
            raise ValueError("n must be positive")
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=False,
            truncation=True,
            max_length=4000,
        ).to(self.model.device)
        batch = {key: value.repeat(n, 1) for key, value in inputs.items()}
        options = dict(kwargs)
        if temperature > 0:
            options["temperature"] = temperature
        with torch.inference_mode():
            outputs = self.model.generate(
                **batch,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                **options,
            )
        input_length = batch["input_ids"].shape[1]
        responses = []
        for output in outputs:
            response = self.tokenizer.decode(
                output[input_length:], skip_special_tokens=True
            )
            for stop_seq in [stop] if isinstance(stop, str) else stop or []:
                if stop_seq:
                    response = response.split(stop_seq, 1)[0]
            responses.append(response.strip())
        return responses

    def cleanup(self):
        # Clean up GPU memory
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        torch.cuda.empty_cache()
        print("Done!")


async def run_generation(
    service: InferenceService,
    model: str,
    prompt: str,
    prompt_paraphrases: list[str] | None,
    num_generations: int,
    sampling: str,
    max_retries: int = 10,
    **gen_kwargs,
) -> list[str]:
    """`gen_kwargs` (max_tokens, temperature, reasoning_effort) go to the service."""
    if num_generations < 1 or max_retries < 1:
        raise ValueError("Generation count and retry count must be positive")
    if sampling not in {"regenerate", "in-context", "paraphrase", "system-prompt"}:
        raise ValueError("Unknown sampling method " + sampling)
    if sampling == "paraphrase" and (
        not prompt_paraphrases
        or len(prompt_paraphrases) != num_generations
        or any(not isinstance(p, str) or not p.strip() for p in prompt_paraphrases)
    ):
        raise ValueError(
            "Paraphrase sampling requires one nonempty paraphrase per generation"
        )
    responses = []
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            if sampling == "regenerate":
                # parallel generation w/o context
                responses = await service.generate(
                    model=model, messages=messages, n=num_generations, **gen_kwargs
                )

            elif sampling == "in-context":
                while len(responses) < num_generations:
                    response = await service.generate(
                        model=model, messages=messages, **gen_kwargs
                    )
                    validate_generations(response, 1)
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
                while len(responses) < num_generations:
                    messages = [
                        {"role": "user", "content": prompt_paraphrases[len(responses)]}
                    ]
                    response = await service.generate(
                        model=model, messages=messages, **gen_kwargs
                    )
                    validate_generations(response, 1)
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
                    model=model, messages=messages, n=num_generations, **gen_kwargs
                )
            else:
                raise Exception("Unknown mode " + sampling)

            validate_generations(responses, num_generations)
            return responses

        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(
                    f"Error generating response for prompt '{prompt}' after {max_retries} attempts: {e}",
                    flush=True,
                )
                raise RuntimeError(
                    f"Generation failed after {max_retries} attempts"
                ) from e

            # Exponential backoff
            wait_time = min(5 * 2**attempt, 60)  # 5, 10, 20, 40, 60, 60, ... seconds
            print(
                f"Attempt {attempt + 1} failed ({e!r:.200}), retrying in {wait_time} seconds...",
                flush=True,
            )
            await asyncio.sleep(wait_time)


def generation_config(
    model, mode, sampling, num_generations, max_tokens, temperature, reasoning_effort
):
    """Everything that defines a generation protocol; part of the cache key."""
    return {
        "stage": "generation",
        "version": 2,
        "model": model,
        "mode": mode,
        "sampling": sampling,
        "num_generations": num_generations,
        "temperature": None if reasoning_effort else temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }


def validate_generations(generations, expected):
    if not isinstance(generations, list) or len(generations) != expected:
        raise ValueError(f"Expected {expected} responses")
    if any(not isinstance(text, str) or not text.strip() for text in generations):
        raise ValueError("Every response must contain nonempty text")


async def process_prompts(
    prompts,
    service,
    model,
    output_file,
    num_generations,
    concurrent_requests,
    sampling,
    mode=None,
    max_tokens=512,
    temperature=1.0,
    reasoning_effort=None,
):
    """Resume matching generations; keep the previous output on any failure."""
    if num_generations < 1 or concurrent_requests < 1:
        raise ValueError("Generation count and concurrency must be positive")
    prompts = list(prompts)
    config = generation_config(
        model,
        mode or type(service).__name__,
        sampling,
        num_generations,
        max_tokens,
        temperature,
        reasoning_effort,
    )
    gen_kwargs = {"max_tokens": max_tokens, "temperature": temperature}
    if reasoning_effort:
        gen_kwargs["reasoning_effort"] = reasoning_effort

    async def compute(prompt):
        generations = await run_generation(
            service,
            model,
            prompt["prompt"],
            prompt.get("prompt_paraphrases"),
            num_generations,
            sampling,
            **gen_kwargs,
        )
        validate_generations(generations, num_generations)
        return {"model": model, "generations": generations}

    await run_cached(
        prompts, output_file, "generation_key", config, compute, concurrent_requests
    )


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
            "anthropic-vertex",
            "vertex",
            "deepseek",
            "transformers",
        ],
        required=True,
        help="Inference service provider (vllm for local server, openai for API, transformers for local HF models, etc.)",
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
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        help="reasoning models: sets the effort and drops sampling parameters",
    )
    parser.add_argument(
        "--limit", type=int, help="only the first N prompts (smoke tests)"
    )
    parser.add_argument("--project", help="Google Cloud project for Vertex providers")
    parser.add_argument("--region", help="Google Cloud region for Vertex providers")
    args = parser.parse_args()
    if args.num_generations < 1 or args.concurrent_requests < 1:
        parser.error("--num-generations and --concurrent-requests must be positive")

    dataset = load_dataset("yimingzhang/novelty-bench", split=args.data)
    if args.limit:
        dataset = dataset.select(range(args.limit))
    eval_dir = (
        args.eval_dir if args.eval_dir else os.path.join(f"{args.data}-evals", args.model)
    )
    os.makedirs(eval_dir, exist_ok=True)
    output_file = os.path.join(eval_dir, "generations.jsonl")

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
    elif args.mode == "anthropic-vertex":
        service = AnthropicVertexService(args.project, args.region or "us-east5")
    elif args.mode == "vertex":
        service = VertexService(args.project, args.region or "us-central1")
    elif args.mode == "deepseek":
        service = DeepSeekService()
    elif args.mode == "transformers":
        service = TransformersService(args.model)
        # Reduce concurrent requests for local inference to avoid memory issues
        concurrent_requests = 1
    else:
        raise Exception(f"unknown service {args.mode}")
    try:
        await process_prompts(
            dataset,
            service,
            args.model,
            output_file,
            args.num_generations,
            concurrent_requests,
            args.sampling,
            mode=args.mode,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )

    finally:
        service.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
