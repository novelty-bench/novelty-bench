"""Check historical result compatibility without modifying released artifacts."""

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
        legacy = {
            "2025-03-27_gemini-1.5-pro",
            "2025-11-21_CrPO-llama-3.1-8b-instruct-cre",
        }
        count = 0
        for path in paths:
            with self.subTest(path=str(path)):
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                stored = json.loads(path.with_name("summary.json").read_text())
                original = summarize(pd.DataFrame(rows))
                corrected = summarize(
                    pd.DataFrame(
                        [dict(row, distinct=len(set(row["partition"]))) for row in rows]
                    )
                )
                self.assertEqual(original, corrected)
                offset = 1 if path.parent.parent.name in legacy else 0
                self.assertAlmostEqual(
                    original["mean_distinct"], stored["mean_distinct"] + offset
                )
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
                    self.assertEqual(
                        len(set(row["partition"])), len(row["partition_scores"])
                    )
                count += len(rows)
        self.assertEqual(count, 4400)
