import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

from src import partition, score
from src.evaluation_io import cached_rows, write_rows

CONFIG = {"stage": "test"}


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

    async def test_pairwise_partition_joins_first_matching_head(self):
        async def same_first_letter(prompt, a, b):
            return a[0] == b[0]

        got = await partition.partition_pairwise(
            "p", ["ax", "bx", "ay", "by"], same_first_letter
        )
        self.assertEqual(got, [0, 1, 0, 1])

    async def test_llm_partition_unshuffles_and_retries_invalid(self):
        outputs = [
            partition.Partition(groups=[[0, 1]]),  # drops index 2: invalid
            partition.Partition(groups=[[2], [0, 1]]),
        ]
        with patch.object(partition, "judge", AsyncMock(side_effect=outputs)) as j:
            got = await partition.partition_llm("p", ["a", "b", "c"], "m", seed=1)
        self.assertEqual(j.await_count, 2)
        self.assertEqual(sorted(got), [0, 0, 1])
        self.assertEqual(got[0], 0)  # canonical: first response is class 0

    async def test_partition_cache_and_failure(self):
        row = {"id": "a", "prompt": "p", "generations": ["a", "b"]}
        alg = AsyncMock(return_value=[0, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "partitions.jsonl")
            await partition.process_instances([row], path, alg, CONFIG)
            self.assertEqual(cached_rows(path)["a"]["distinct"], 2)
            await partition.process_instances([row], path, alg, CONFIG)
            self.assertEqual(alg.await_count, 1)
            alg.side_effect = RuntimeError("unavailable")
            before = Path(path).read_bytes()
            with self.assertRaises(RuntimeError):
                await partition.process_instances(
                    [dict(row, prompt="changed")], path, alg, CONFIG
                )
            self.assertEqual(Path(path).read_bytes(), before)

    async def test_journal_resumes_after_partial_failure(self):
        rows = [{"id": i, "prompt": i, "generations": ["a"]} for i in "xy"]

        async def flaky(prompt, generations):
            if prompt == "y":
                raise RuntimeError("unavailable")
            return [0]

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "partitions.jsonl")
            with self.assertRaises(RuntimeError):
                await partition.process_instances(rows, path, flaky, CONFIG)
            self.assertFalse(Path(path).exists())
            self.assertEqual(set(cached_rows(path + ".partial")), {"x"})
            alg = AsyncMock(return_value=[0])
            await partition.process_instances(rows, path, alg, CONFIG)
            self.assertEqual(alg.await_count, 1)  # x came from the journal
            self.assertEqual(set(cached_rows(path)), {"x", "y"})
            self.assertFalse(Path(path + ".partial").exists())

    async def test_score_cache_invalidated_by_partition_and_patience(self):
        row = {"id": "a", "prompt": "p", "generations": ["a", "b"], "partition": [0, 1]}
        scorer = AsyncMock(return_value=[5, 10])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "scores.jsonl")
            await score.process_instances([row], path, scorer, CONFIG, 0.8)
            await score.process_instances([row], path, scorer, CONFIG, 0.8)
            self.assertEqual(scorer.await_count, 1)
            await score.process_instances(
                [dict(row, partition=[0, 0])], path, scorer, CONFIG, 0.8
            )
            self.assertEqual(cached_rows(path)["a"]["generation_scores"], [5, 10])
            self.assertEqual(cached_rows(path)["a"]["partition_scores"], [5])
            await score.process_instances([row], path, scorer, CONFIG, 0)
            self.assertEqual(cached_rows(path)["a"]["utility"], 5)
            self.assertEqual(scorer.await_count, 3)

    async def test_legacy_cache_does_not_skip_new_evaluation(self):
        row = {"id": "a", "prompt": "p", "generations": ["a"], "partition": [0]}
        scorer = AsyncMock(return_value=[5])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "scores.jsonl")
            write_rows(path, [dict(row, utility=10)])
            await score.process_instances([row], path, scorer, CONFIG, 0.8)
            self.assertEqual(scorer.await_count, 1)
            self.assertEqual(cached_rows(path)["a"]["utility"], 5)

    async def test_llm_scores_cap_invalid_responses(self):
        out = score.Scores(
            items=[
                score.Verdict(valid=True, score=9),
                score.Verdict(valid=False, score=8),
            ]
        )
        with patch.object(score, "judge", AsyncMock(return_value=out)):
            self.assertEqual(await score.score_llm("p", ["a", "b"], "m"), [9, 3])
