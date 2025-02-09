import argparse
import json
import os

import numpy as np
import pandas as pd

from src.common import DATASETS


def summarize(df: pd.DataFrame, decay_rate=0.5) -> dict:
    all_scores = []
    summary = {}

    raw_quality_at_k_arr = [[] for _ in range(10)]

    for _, row in df.iterrows():
        scores = row["partition_scores"]
        for i, s in enumerate(scores):
            raw_quality_at_k_arr[i].append(s)

        scores = scores + [0] * (10 - len(scores))
        all_scores.append(scores)
    all_scores = np.array(all_scores)

    quality_at_k_arr = []
    for k in range(10):
        quality_at_k = all_scores[:, k].mean()
        quality_at_k_arr.append(quality_at_k.item())

    counts = df["partition"].map(len).value_counts()
    summary["mean-equivalence-classes"] = df["partition"].map(len).mean()
    summary["equivalence-classes-histogram"] = [
        int(counts.get(c, 0)) for c in range(1, 11)
    ]
    summary["raw-quality-at-k"] = [
        np.mean(r).item() if r else 0.0 for r in raw_quality_at_k_arr
    ]
    summary["quality-at-k"] = quality_at_k_arr
    rbq = np.average(quality_at_k_arr, weights=decay_rate ** np.arange(10))
    summary["rank-biased-quality"] = rbq.item()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="curated", choices=DATASETS)
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    df = pd.read_json(os.path.join(eval_dir, "scores.jsonl"), lines=True)
    summary = summarize(df)
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
