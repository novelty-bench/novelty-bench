import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src import batch_generate, inference
from src.evaluation_io import cached_rows

PROMPTS = [{"id": "a", "prompt": "p"}, {"id": "b", "prompt": "q"}]
CONFIG = inference.generation_config("gpt-x", "openai", "regenerate", 2, 2048, 1.0, "low")


def completion(text, finish="stop"):
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-x",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {"role": "assistant", "content": text},
            }
        ],
    }


class OpenAIBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_and_collect(self):
        uploaded = {}
        files = SimpleNamespace(
            create=lambda file, purpose: (
                uploaded.update(body=file[1].read()) or SimpleNamespace(id="f1")
            ),
            content=lambda fid: SimpleNamespace(text=self.output),
        )
        batches = SimpleNamespace(
            create=lambda **kw: SimpleNamespace(id="b1"),
            retrieve=lambda bid: SimpleNamespace(
                id=bid, status="completed", output_file_id="o1", request_counts={}
            ),
        )
        client = SimpleNamespace(files=files, batches=batches)
        # a: both samples fine; b: sample 1 is a content filter, sample 0 missing
        self.output = "\n".join(
            json.dumps(r)
            for r in [
                {
                    "custom_id": "c0",
                    "response": {"status_code": 200, "body": completion("a0")},
                },
                {
                    "custom_id": "c1",
                    "response": {"status_code": 200, "body": completion("a1")},
                },
                {
                    "custom_id": "c3",
                    "response": {
                        "status_code": 200,
                        "body": completion(None, "content_filter"),
                    },
                },
            ]
        )
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(batch_generate.openai, "OpenAI", return_value=client),
        ):
            batch_generate.submit(d, PROMPTS, CONFIG)
            lines = [json.loads(line) for line in uploaded["body"].decode().splitlines()]
            self.assertEqual([x["custom_id"] for x in lines], ["c0", "c1", "c2", "c3"])
            self.assertEqual(lines[0]["body"]["reasoning_effort"], "low")
            self.assertEqual(lines[0]["body"]["max_completion_tokens"], 2048)
            self.assertNotIn("temperature", lines[0]["body"])

            live = AsyncMock(return_value=["b0 live"])
            with patch.object(batch_generate, "OpenAIService") as svc:
                svc.return_value.generate = live
                await batch_generate.collect(d, PROMPTS)
            live.assert_awaited_once()
            rows = cached_rows(str(Path(d) / "generations.jsonl"))
            self.assertEqual(rows["a"]["generations"], ["a0", "a1"])
            self.assertEqual(rows["b"]["generations"], ["b0 live", inference.REFUSED])
            self.assertEqual(rows["b"]["generation_config"], CONFIG)
            self.assertFalse((Path(d) / batch_generate.MANIFEST).exists())

    async def test_live_and_batch_rows_share_cache_keys(self):
        """A live run after a batch collect must see every row as cached."""
        row = {"id": "a", "prompt": "p"}
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "generations.jsonl")
            service = SimpleNamespace(generate=AsyncMock(return_value=["x", "y"]))
            await inference.process_prompts(
                [row],
                service,
                "gpt-x",
                path,
                2,
                1,
                "regenerate",
                mode="openai",
                max_tokens=2048,
                reasoning_effort="low",
            )
            self.assertEqual(
                cached_rows(path)["a"]["generation_key"],
                batch_generate.plan([row], path, "generation_key", CONFIG)[1]["a"],
            )


class ParamTests(unittest.TestCase):
    def test_anthropic_reasoning_params(self):
        body = inference.anthropic_params(
            "claude-x", [{"role": "user", "content": "hi"}], 2048, 1.0, "low"
        )
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertEqual(body["output_config"], {"effort": "low"})
        self.assertNotIn("temperature", body)
        self.assertEqual(
            body["messages"][-1]["content"][0]["cache_control"], {"type": "ephemeral"}
        )

    def test_anthropic_refusal_placeholder(self):
        msg = SimpleNamespace(stop_reason="refusal", content=[])
        self.assertEqual(inference.anthropic_text(msg), inference.REFUSED)
        msg = SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")]
        )
        self.assertEqual(inference.anthropic_text(msg), "ok")

    def test_openai_params_without_reasoning_keep_sampling(self):
        body = inference.openai_params(
            "gpt-4o", [{"role": "user", "content": "hi"}], 512, 1.0
        )
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["temperature"], 1.0)
