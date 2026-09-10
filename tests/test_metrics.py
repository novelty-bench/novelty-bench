import tempfile
import unittest
from pathlib import Path

from src.evaluation_io import cached_rows, evaluation_key, write_rows
from src.score import score_first_occurrences


class MetricsTests(unittest.TestCase):
    def test_duplicates_only_receive_first_occurrence_credit(self):
        credited, classes = score_first_occurrences([5, 10, 8], [0, 0, 1])
        self.assertEqual(credited, [5, 0, 8])
        self.assertEqual(classes, [5, 8])

    def test_cache_tracks_inputs_and_configuration(self):
        row = {"id": "a", "prompt": "p", "generations": ["a"], "partition": [0]}
        key = evaluation_key(row, {"patience": 0.8})
        for changed in [
            dict(row, prompt="q"),
            dict(row, generations=["b"]),
            dict(row, partition=[1]),
        ]:
            self.assertNotEqual(key, evaluation_key(changed, {"patience": 0.8}))
        self.assertNotEqual(key, evaluation_key(row, {"patience": 0.9}))

    def test_failed_write_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "scores.jsonl")
            write_rows(path, [{"id": "original"}])

            def interrupted():
                yield {"id": "new"}
                raise RuntimeError("interrupted")

            with self.assertRaises(RuntimeError):
                write_rows(path, interrupted())
            self.assertEqual(cached_rows(path), {"original": {"id": "original"}})
