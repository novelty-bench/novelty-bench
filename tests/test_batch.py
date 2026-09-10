import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src import batch, partition
from src.evaluation_io import cached_rows

CONFIG = {
    "stage": "partition",
    "version": "1.1",
    "judge_model": "m",
    "effort": "high",
    "alg": "llm",
    "seed": None,
}


def message(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)]
    )


def result(custom_id, msg=None, kind="succeeded"):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type=kind, message=msg),
        model_dump=lambda mode=None: {"custom_id": custom_id, "type": kind},
    )


class FakeBatches:
    def __init__(self, results):
        self._results = results
        self.created = None

    def create(self, requests):
        self.created = requests
        return SimpleNamespace(id="b1")

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended", request_counts={})

    def results(self, batch_id):
        return iter(self._results)


class BatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_then_collect_with_live_retry(self):
        rows = [
            {"id": "a b", "prompt": "p", "generations": ["x", "y"]},
            {"id": "c", "prompt": "q", "generations": ["x", "y"]},
        ]
        results = [
            result("c0", message('{"groups": [[0], [1]]}')),  # a b: valid
            result("c1", message('{"groups": [[0]]}')),  # c: drops index 1 -> live retry
        ]
        client = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches(results)))
        with tempfile.TemporaryDirectory() as directory:
            eval_dir = str(Path(directory))
            vdir = str(Path(directory) / "v1.1")
            Path(vdir).mkdir()
            with open(Path(eval_dir) / "generations.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            batch.submit(client, partition.STAGE, eval_dir, vdir, CONFIG)
            self.assertEqual(
                [r["custom_id"] for r in client.messages.batches.created], ["c0", "c1"]
            )
            manifest = json.loads((Path(vdir) / "batch.partition.json").read_text())
            self.assertEqual(manifest["calls"], [["a b", 0], ["c", 0]])

            live = AsyncMock(return_value={"partition": [0, 0], "distinct": 1})
            with patch.object(batch, "run_stage", live):
                await batch.collect(client, partition.STAGE, eval_dir, vdir)
            live.assert_awaited_once()
            out = cached_rows(str(Path(vdir) / "partitions.jsonl"))
            self.assertEqual(out["a b"]["partition"], [0, 1])
            self.assertEqual(out["c"]["partition"], [0, 0])
            self.assertEqual(out["c"]["partition_config"], CONFIG)
            self.assertFalse((Path(vdir) / "batch.partition.json").exists())
            self.assertFalse((Path(vdir) / "partitions.jsonl.partial").exists())

    async def test_refusal_is_not_parsed(self):
        from src.judge import parse_message

        with self.assertRaises(ValueError):
            parse_message(message("", stop_reason="refusal"), partition.Partition)
