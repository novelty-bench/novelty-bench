import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from transformers import BatchEncoding

from src import common, inference


class TransformersTests(unittest.TestCase):
    def make_service(self):
        service = inference.TransformersService.__new__(inference.TransformersService)
        service.model = Mock(device="cpu")
        service.tokenizer = Mock(pad_token_id=0, eos_token_id=3)
        service.tokenizer.apply_chat_template.return_value = BatchEncoding(
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }
        )
        service.model.generate.side_effect = lambda **kw: torch.cat(
            [kw["input_ids"], torch.full((kw["input_ids"].shape[0], 1), 3)], dim=1
        )
        service.tokenizer.decode.return_value = "answer STOP ignored"
        return service

    def test_single_batch_and_greedy_share_valid_tensor_path(self):
        for n, temperature in [(1, 1), (3, 1), (3, 0)]:
            with self.subTest(n=n, temperature=temperature):
                service = self.make_service()
                responses = service._generate_sync(
                    [{"role": "user", "content": "p"}], n, 10, temperature, ["STOP"], {}
                )
                self.assertEqual(responses, ["answer"] * n)
                kwargs = service.model.generate.call_args.kwargs
                self.assertEqual(kwargs["input_ids"].shape, (n, 2))
                self.assertEqual(kwargs["do_sample"], temperature > 0)
                self.assertEqual("temperature" in kwargs, temperature > 0)
                for call in service.tokenizer.decode.call_args_list:
                    self.assertTrue(call.kwargs["skip_special_tokens"])

    def test_eager_fallback_does_not_pass_stop_to_loader(self):
        model = Mock(device="cpu")
        model.get_memory_footprint.return_value = 0
        with (
            patch.object(inference.AutoTokenizer, "from_pretrained"),
            patch.object(
                inference.AutoModelForCausalLM,
                "from_pretrained",
                side_effect=[ImportError("no flash attention"), model],
            ) as loader,
        ):
            inference.TransformersService("test")
        self.assertEqual(loader.call_args.kwargs["attn_implementation"], "eager")
        self.assertNotIn("stop", loader.call_args.kwargs)


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_results_and_failures_raise(self):
        for output in [[], [None, None], ["", ""], ["a"]]:
            service = SimpleNamespace(generate=AsyncMock(return_value=output))
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                await inference.run_generation(
                    service, "m", "p", None, 2, "regenerate", max_retries=1
                )
        service = SimpleNamespace(generate=AsyncMock(side_effect=RuntimeError("down")))
        with self.assertRaises(RuntimeError):
            await inference.run_generation(
                service, "m", "p", None, 1, "regenerate", max_retries=1
            )

    async def test_in_context_and_paraphrase(self):
        service = SimpleNamespace(generate=AsyncMock(side_effect=[["a"], ["b"]]))
        result = await inference.run_generation(
            service, "m", "p", None, 2, "in-context", max_retries=1
        )
        self.assertEqual(result, ["a", "b"])
        service.generate = AsyncMock(side_effect=[["a"], ["b"]])
        self.assertEqual(
            await inference.run_generation(
                service, "m", "p", ["p1", "p2"], 2, "paraphrase", max_retries=1
            ),
            ["a", "b"],
        )

    async def test_generation_cache_invalidation_and_failure_preservation(self):
        row = {"id": "x", "prompt": "p"}
        service = SimpleNamespace(generate=AsyncMock(return_value=["a"]))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "generations.jsonl")

            async def run(prompt=row, model="m", sampling="regenerate"):
                await inference.process_prompts(
                    [prompt], service, model, path, 1, 1, sampling, mode="test"
                )

            await run()
            await run()
            self.assertEqual(service.generate.await_count, 1)
            await run(model="new-model")
            await run(prompt=dict(row, prompt="new-prompt"))
            await run(sampling="system-prompt")
            self.assertEqual(service.generate.await_count, 4)
            before = Path(path).read_bytes()
            with (
                patch.object(
                    inference,
                    "run_generation",
                    AsyncMock(side_effect=RuntimeError("down")),
                ),
                self.assertRaises(RuntimeError),
            ):
                await run()
            self.assertEqual(Path(path).read_bytes(), before)
            self.assertEqual(
                json.loads(before)["generation_config"]["sampling"], "system-prompt"
            )

    async def test_invalid_concurrency_rejected_before_work(self):
        with self.assertRaises(ValueError):
            await asyncio.wait_for(
                inference.process_prompts([], None, "m", "unused", 1, 0, "regenerate"),
                timeout=1,
            )

    async def test_gemini_system_and_multiblock_text(self):
        service = inference.GeminiService.__new__(inference.GeminiService)

        def part(text, thought=False):
            return SimpleNamespace(text=text, thought=thought)

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[part("internal", True), part("hello "), part("world")]
                    )
                )
            ]
        )
        call = AsyncMock(return_value=response)
        service.client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=call))
        )
        result = await service.generate(
            "model",
            [
                {"role": "system", "content": "instruction"},
                {"role": "user", "content": "prompt"},
            ],
        )
        self.assertEqual(result, ["hello world"])
        self.assertEqual(
            call.call_args.kwargs["config"].system_instruction, "instruction"
        )
        self.assertEqual([x.role for x in call.call_args.kwargs["contents"]], ["user"])
        for candidates in [
            [],
            [SimpleNamespace(content=None)],
            [SimpleNamespace(content=SimpleNamespace(parts=[]))],
        ]:
            call.return_value = SimpleNamespace(candidates=candidates)
            with self.assertRaises(ValueError):
                await service.generate("model", [{"role": "user", "content": "prompt"}])


class ProviderConfigurationTests(unittest.TestCase):
    def test_openai_environment_and_file_priority(self):
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-key"}),
            patch.object(common, "AsyncOpenAI") as factory,
        ):
            common.oai_client()
            factory.assert_called_once_with()

    def test_vertex_url_and_project(self):
        credentials = Mock(token="synthetic-key")
        with (
            patch.object(
                inference, "default", return_value=(credentials, "default-project")
            ),
            patch.object(inference.transport.requests, "Request"),
            patch.dict(os.environ, {}, clear=True),
        ):
            service = inference.VertexService()
            self.assertEqual(
                str(service.client._prepare_url("/chat/completions")),
                "https://us-central1-aiplatform.googleapis.com/v1/projects/default-project/locations/us-central1/endpoints/openapi/chat/completions",
            )
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            common.google_project()
        self.assertEqual(common.google_project("explicit"), "explicit")
