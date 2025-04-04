import numpy as np
import argparse
import json
import math
import os
from collections import deque
from contextlib import nullcontext

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

parser = argparse.ArgumentParser()

OUTPUT_DIR = "models/similarity-classifier"
TRAIN_FILE = "data/classifier/train.jsonl"
VAL_FILE = "data/classifier/val.jsonl"
LABEL_COLUMN = "similar"
WARMUP_STEPS = 10
TRAIN_STEPS = 80
PRETRAINED_MODEL = "microsoft/deberta-v3-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
VAL_BATCH_SIZE = 16
MAX_LR = 2e-5
MIN_LR = 2e-6
MAX_LEN = 128
GRAD_ACC_STEPS = 8
AUTOCAST = (
    torch.autocast(
        DEVICE,
        dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
    )
    if DEVICE == "cuda"
    else nullcontext()
)


def get_lr(t: int) -> float:
    assert MAX_LR >= MIN_LR >= 0.0
    assert TRAIN_STEPS >= WARMUP_STEPS >= 0

    if t <= WARMUP_STEPS:
        return (t / WARMUP_STEPS) * MAX_LR

    elif t >= TRAIN_STEPS:
        return MIN_LR

    return (MAX_LR - MIN_LR) / 2 * math.cos(
        (t - WARMUP_STEPS) * math.pi / (TRAIN_STEPS - WARMUP_STEPS)
    ) + (MIN_LR + MAX_LR) / 2


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def hyperparameters():
    return {
        key: value
        for key, value in globals().items()
        if key
        in [
            "TRAIN_FILE",
            "VAL_FILE",
            "LABEL_COLUMN",
            "TRAIN_STEPS",
            "WARMUP_STEPS",
            "PRETRAINED_MODEL",
            "BATCH_SIZE",
            "MAX_LR",
            "MIN_LR",
            "MAX_LEN",
            "GRAD_ACC_STEPS",
        ]
    }


def to_device(data):
    return {k: v.to(DEVICE) for k, v in data.items()}


class DictDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        return self.data[i]


def get_dataloader(tokenizer, df, train):
    data = []
    for _, row in tqdm(df.iterrows(), "preping data..."):
        prompt = row["prompt"]
        generation_0 = row["generation_0"]
        generation_1 = row["generation_1"]
        input_ids = [tokenizer.cls_token_id]
        for s in [generation_0, generation_1]:
            input_ids.extend(
                tokenizer.encode(
                    s,
                    truncation=True,
                    max_length=MAX_LEN,
                    add_special_tokens=False,
                )
            )
            input_ids.append(tokenizer.sep_token_id)
            prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
        token_type_ids = [0] * prompt_len + [1] * (
            len(input_ids) - prompt_len
        )

        data.append(
            {
                "input_ids": torch.LongTensor(input_ids),
                "token_type_ids": torch.LongTensor(token_type_ids),
                "labels": row[LABEL_COLUMN],
            }
        )

        if train:
            input_ids = [tokenizer.cls_token_id]
            for s in [generation_1, generation_0]:
                input_ids.extend(
                    tokenizer.encode(
                        s,
                        truncation=True,
                        max_length=MAX_LEN,
                        add_special_tokens=False,
                    )
                )
                input_ids.append(tokenizer.sep_token_id)
                prompt_len = input_ids.index(tokenizer.sep_token_id) + 1
            token_type_ids = [0] * prompt_len + [1] * (
                len(input_ids) - prompt_len
            )
            
            data.append(
                {
                    "input_ids": torch.LongTensor(input_ids),
                    "token_type_ids": torch.LongTensor(token_type_ids),
                    "labels": row[LABEL_COLUMN],
                }
            )

    dataset = DictDataset(data)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE if train else VAL_BATCH_SIZE,
        shuffle=train,
        pin_memory=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


def get_train_iter(dl):
    while True:
        for batch in dl:
            yield to_device(batch)


def main():
    model = AutoModelForSequenceClassification.from_pretrained(
        "deberta-v3-large-generation-similarity",
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    val_data = pd.read_json(VAL_FILE, lines=True)
    val_dl = get_dataloader(tokenizer, val_data, False)

    model.eval()
    labels = val_data[LABEL_COLUMN].tolist()
    preds = []
    with torch.inference_mode():
        for batch in tqdm(val_dl, total=math.ceil(len(val_data) / VAL_BATCH_SIZE)):
            batch = to_device(batch)
            with AUTOCAST:
                outputs = model(**batch)
            preds_batch = outputs["logits"].softmax(-1)[:, 1].tolist()
            preds.extend([1 if p > 0.102 else 0 for p in preds_batch])
        
    print(len(labels), len(preds))
    assert len(labels) == len(preds)


    val_data[LABEL_COLUMN + "-pred"] = preds

    val_eval = {}
    val_eval["precision"] = precision_score(labels, preds)
    val_eval["recall"] = recall_score(labels, preds)
    val_eval["f1"] = f1_score(labels, preds)
    val_eval["accuracy"] = accuracy_score(labels, preds)

    print(json.dumps(val_eval, indent=2))

if __name__ == "__main__":
    main()