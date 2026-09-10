import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

from src import partition, score
from src.evaluation_io import cached_rows, write_rows


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifier_disables_gradients_inside_coroutine(self):
        tokenizer = SimpleNamespace(
            cls_token_id=1, sep_token_id=2, encode=lambda *a, **k: [3]
        )

        def model(**kwargs):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
            return {"logits": torch.tensor([[0.0, 1.0]])}

        with (
            patch.object(
                partition,
                "load_deberta_tokenizer_and_model",
                return_value=(tokenizer, model),
            ),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            await partition.classifier_score("p", "a", "b")

    async def test_bertscore_scalar(self):
        with patch.object(partition, "bertscore", AsyncMock(return_value=0.8)):
            self.assertTrue(await partition.equivalence_check_bertscore("p", "a", "b"))

    async def test_short_answers_are_not_merged_by_word_overlap(self):
        self.assertIsNone(partition.maybe_test_equality("do it", "do not do it"))
        self.assertTrue(partition.maybe_test_equality("yes", "yes"))

    async def test_judge_failure_propagates(self):
        parse = AsyncMock(side_effect=RuntimeError("unavailable"))
        client = SimpleNamespace(
            beta=SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
            )
        )
        with (
            patch.object(partition, "load_judge_client", return_value=client),
            self.assertRaises(RuntimeError),
        ):
            await partition.equivalence_check_gpt4("p", "a", "b")

    async def test_partition_cache_and_failure(self):
        row = {"id": "a", "prompt": "p", "generations": ["a", "b"]}
        checker = AsyncMock(return_value=False)
        checker.__name__ = "test_checker"
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "partitions.jsonl")
            await partition.process_instances([row], path, checker)
            self.assertEqual(cached_rows(path)["a"]["distinct"], 2)
            await partition.process_instances([row], path, checker)
            self.assertEqual(checker.await_count, 1)
            checker.side_effect = RuntimeError("unavailable")
            before = Path(path).read_bytes()
            with self.assertRaises(RuntimeError):
                await partition.process_instances(
                    [dict(row, prompt="changed")], path, checker
                )
            self.assertEqual(Path(path).read_bytes(), before)

    async def test_score_cache_invalidated_by_partition_and_patience(self):
        row = {"id": "a", "prompt": "p", "generations": ["a", "b"], "partition": [0, 1]}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                score, "score_partition_rm", AsyncMock(return_value=([5, 10], [5, 10]))
            ) as rm,
        ):
            path = str(Path(directory) / "scores.jsonl")
            await score.process_instances([row], path, 0.8)
            await score.process_instances([row], path, 0.8)
            self.assertEqual(rm.await_count, 1)
            rm.return_value = ([5, 0], [5])
            await score.process_instances([dict(row, partition=[0, 0])], path, 0.8)
            self.assertEqual(cached_rows(path)["a"]["generation_scores"], [5, 0])
            rm.return_value = ([5, 10], [5, 10])
            await score.process_instances([row], path, 0)
            self.assertEqual(cached_rows(path)["a"]["utility"], 5)
            self.assertEqual(rm.await_count, 3)

    async def test_legacy_cache_does_not_skip_new_evaluation(self):
        row = {"id": "a", "prompt": "p", "generations": ["a"], "partition": [0]}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                score, "score_partition_rm", AsyncMock(return_value=([5], [5]))
            ) as rm,
        ):
            path = str(Path(directory) / "scores.jsonl")
            write_rows(path, [dict(row, utility=10)])
            await score.process_instances([row], path, 0.8)
            self.assertEqual(rm.await_count, 1)
            self.assertEqual(cached_rows(path)["a"]["utility"], 5)
