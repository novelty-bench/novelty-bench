import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src import publish
from src.evaluation_io import cached_rows
from src.slim import slim_file, slim_row
from src.summarize import summarize

SCORE_ROW = {
    "id": "a",
    "prompt": "p",
    "model": "m",
    "generations": ["long response one", "long response two"],
    "partition": [0, 0],
    "distinct": 1,
    "generation_scores": [7, None],
    "partition_scores": [7],
    "utility": 3.888,
    "unscored": "judge_refusal",
    "partition_key": "k1",
    "score_key": "k2",
    "partition_config": {"stage": "partition"},
    "score_config": {"stage": "score", "patience": 0.8},
}


class SlimTests(unittest.TestCase):
    def test_drops_only_response_text(self):
        slim = slim_row(SCORE_ROW)
        self.assertNotIn("generations", slim)
        for field in SCORE_ROW:
            if field != "generations":
                self.assertEqual(slim[field], SCORE_ROW[field], field)

    def test_slim_file_rewrites_in_place_and_shrinks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "scores.jsonl")
            with open(path, "w") as f:
                for i in range(3):
                    f.write(json.dumps(dict(SCORE_ROW, id=f"a{i}")) + "\n")
            before = Path(path).stat().st_size
            self.assertEqual(slim_file(path), 3)
            self.assertLess(Path(path).stat().st_size, before)
            rows = cached_rows(path)
            self.assertEqual(set(rows), {"a0", "a1", "a2"})
            self.assertNotIn("generations", rows["a0"])

    def test_summarize_still_works_on_slim_rows(self):
        rows = [slim_row(dict(SCORE_ROW, id=f"a{i}")) for i in range(4)]
        summary = summarize(pd.DataFrame(rows), "1.1")
        self.assertEqual(summary["version"], "1.1")
        self.assertAlmostEqual(summary["mean_distinct"], 1.0)
        self.assertAlmostEqual(summary["mean_utility"], 3.888)

    def test_out_leaves_the_original_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "partitions.jsonl"
            src.write_text(json.dumps(SCORE_ROW) + "\n")
            out = str(Path(directory) / "slim.jsonl")
            slim_file(str(src), out)
            self.assertIn("generations", json.loads(src.read_text()))
            self.assertNotIn("generations", json.loads(Path(out).read_text()))


class PublishTests(unittest.TestCase):
    def test_repo_path_mirrors_the_eval_dir(self):
        self.assertEqual(
            publish.repo_path("2026-09-10_m", "nb-curated", "generations.jsonl"),
            "results/2026-09-10_m/nb-curated/generations.jsonl",
        )
        self.assertEqual(publish.run_name("evaluation/2026-09-10_m/"), "2026-09-10_m")

    def test_push_sends_the_run_folder_under_its_name(self):
        calls = {}
        api = SimpleNamespace(upload_folder=lambda **kw: calls.update(kw))
        where = publish.push(api, "evaluation/2026-09-10_m", "org/ds")
        self.assertEqual(where, "results/2026-09-10_m")
        self.assertEqual(calls["path_in_repo"], "results/2026-09-10_m")
        self.assertEqual(calls["repo_type"], "dataset")
        self.assertEqual(calls["allow_patterns"], publish.PATTERNS)
        # working files never leave the machine
        self.assertFalse(any("batch" in p for p in calls["allow_patterns"]))

    def test_pull_maps_repo_paths_back_onto_the_eval_dir(self):
        remote = [
            "results/2026-09-10_m/nb-curated/generations.jsonl",
            "results/2026-09-10_m/nb-curated/v1.1/scores.jsonl",
            "results/2026-09-10_other/nb-curated/generations.jsonl",
        ]
        api = SimpleNamespace(list_repo_files=lambda **kw: remote)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cached"
            source.write_text("row\n")
            with patch("huggingface_hub.hf_hub_download", return_value=str(source)):
                written = publish.pull(api, "2026-09-10_m", f"{directory}/out", "org/ds")
        names = sorted(w.replace(f"{directory}/out/", "") for w in written)
        self.assertEqual(
            names, ["nb-curated/generations.jsonl", "nb-curated/v1.1/scores.jsonl"]
        )

    def test_pull_rejects_an_unknown_run(self):
        api = SimpleNamespace(list_repo_files=lambda **kw: [])
        with self.assertRaises(SystemExit):
            publish.pull(api, "nope", "/tmp/x", "org/ds")
