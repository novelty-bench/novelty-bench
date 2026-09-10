import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import torch

from src.classifier import evaluate_classifier as evaluator
from src.classifier import finetune_classifier as trainer
from src.data.filter_wildchat_gpt4 import select_prompts
from src.data.process_wildchat import write_splits


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self.forward_modes = []
        self.saved = None

    def forward(self, input_ids, labels):
        self.forward_modes.append(self.training)
        logits = self.linear(input_ids.float())
        return SimpleOutput(logits=logits)

    def save_pretrained(self, path):
        self.saved = path


class SimpleOutput(dict):
    @property
    def logits(self):
        return self["logits"]


class TrainingTests(unittest.TestCase):
    def test_training_mode_and_loadable_checkpoint_contract(self):
        model = TinyClassifier().eval()
        tokenizer = Mock()
        data = pd.DataFrame({"similar": [0, 1]})
        batches = [
            {"input_ids": torch.tensor([[1.0], [2.0]]), "labels": torch.tensor([0, 1])}
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(trainer, "WARMUP_STEPS", 1),
            patch.object(trainer, "TRAIN_STEPS", 1),
            patch.object(trainer, "GRAD_ACC_STEPS", 2),
            patch.object(trainer, "DEVICE", "cpu"),
            patch.object(trainer, "OUTPUT_DIR", directory),
            patch.object(
                trainer.AutoModelForSequenceClassification,
                "from_pretrained",
                return_value=model,
            ),
            patch.object(
                trainer.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ),
            patch.object(trainer.pd, "read_json", return_value=data),
            patch.object(trainer, "get_dataloader", return_value=batches),
        ):
            trainer.main()
            self.assertEqual(model.forward_modes, [True, True, False])
            self.assertEqual(model.saved, directory)
            tokenizer.save_pretrained.assert_called_once_with(directory)
            self.assertTrue((Path(directory) / "eval.json").exists())

    def test_evaluator_uses_same_checkpoint_for_model_and_tokenizer(self):
        model = TinyClassifier()
        batches = [
            {"input_ids": torch.tensor([[1.0], [2.0]]), "labels": torch.tensor([0, 1])}
        ]
        with (
            patch(
                "sys.argv",
                ["evaluate", "--model", "local-checkpoint", "--data", "data/val.jsonl"],
            ),
            patch.object(evaluator, "DEVICE", "cpu"),
            patch.object(
                evaluator.AutoModelForSequenceClassification,
                "from_pretrained",
                return_value=model,
            ) as load_model,
            patch.object(evaluator.AutoTokenizer, "from_pretrained") as load_tokenizer,
            patch.object(
                evaluator.pd, "read_json", return_value=pd.DataFrame({"similar": [0, 1]})
            ),
            patch.object(evaluator, "get_dataloader", return_value=batches),
        ):
            evaluator.main()
            self.assertEqual(load_model.call_args.args[0], "local-checkpoint")
            load_tokenizer.assert_called_once_with("local-checkpoint")

    def test_released_training_paths_exist(self):
        root = Path(__file__).resolve().parents[1]
        for module in (trainer, evaluator):
            self.assertTrue((root / module.TRAIN_FILE).exists())
            self.assertTrue((root / module.VAL_FILE).exists())


class DataTests(unittest.TestCase):
    def test_splits_are_bounded_disjoint_jsonl_and_create_directory(self):
        data = pd.DataFrame(
            {"id": [f"id-{i}" for i in range(5300)], "prompt": ["p"] * 5300}
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "wildchat"
            write_splits(data, output)

            def read(name):
                return [
                    json.loads(line)["id"]
                    for line in (output / name).read_text().splitlines()
                ]

            train, dev, test = (
                read("5k.jsonl"),
                read("dev-no-labels.jsonl"),
                read("test-no-labels.jsonl"),
            )
            self.assertEqual([len(train), len(dev), len(test)], [5000, 100, 100])
            self.assertFalse(set(train) & set(dev))
            self.assertFalse(set(train) & set(test))
            self.assertEqual(train, read("benchmark-no-labels.jsonl"))

    def test_selection_independent_of_completion_order(self):
        data = pd.DataFrame({"id": [str(i) for i in range(20)], "chosen": [True] * 20})
        selected = select_prompts(data, count=10)
        shuffled = select_prompts(data.sample(frac=1, random_state=2), count=10)
        self.assertEqual(list(selected["id"]), list(shuffled["id"]))
        with self.assertRaises(ValueError):
            select_prompts(data, count=21)
