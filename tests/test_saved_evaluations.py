"""Verify historical counts and utility after the one-time distinct correction."""

import json
import unittest
from pathlib import Path

import pandas as pd

from src.summarize import summarize

ROOT = Path(__file__).resolve().parents[1]


class SavedEvaluationTests(unittest.TestCase):
    def test_saved_summaries_and_corrected_fields(self):
        paths = sorted((ROOT / "evaluation").glob("*/*/scores.jsonl"))
        self.assertEqual(len(paths), 8)
        count = 0
        for path in paths:
            with self.subTest(path=str(path)):
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                stored = json.loads(path.with_name("summary.json").read_text())
                original = summarize(pd.DataFrame(rows))
                self.assertAlmostEqual(original["mean_distinct"], stored["mean_distinct"])
                self.assertAlmostEqual(original["mean_utility"], stored["mean_utility"])
                for row in rows:
                    weights = [0.8**i for i in range(len(row["generation_scores"]))]
                    expected = sum(
                        w * score
                        for w, score in zip(
                            weights, row["generation_scores"], strict=True
                        )
                    ) / sum(weights)
                    self.assertAlmostEqual(expected, row["utility"])
                for row in rows:
                    self.assertEqual(row["distinct"], len(set(row["partition"])))
                    self.assertEqual(row["distinct"], len(row["partition_scores"]))
                count += len(rows)
        self.assertEqual(count, 4400)

    def test_saved_partition_counts(self):
        for path in (ROOT / "evaluation").glob("*/*/partitions.jsonl"):
            with self.subTest(path=str(path)):
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    self.assertEqual(row["distinct"], len(set(row["partition"])))
