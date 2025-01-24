import argparse
import json

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame, decay_rate=0.5) -> dict:
    all_scores = []
    summary = {}
    for _, row in df.iterrows():
        scores = row["partition_scores"]
        scores = scores + [0] * (10 - len(scores))
        all_scores.append(scores)
    all_scores = np.array(all_scores)

    quality_at_k_arr = []
    quality_at_k_arr = []
    for k in range(10):
        quality_at_k = all_scores[:, k].mean()
        quality_at_k_arr.append(quality_at_k.item())

    counts = df["partition"].map(len).value_counts()
    summary["mean-equivalence-classes"] = df["partition"].map(len).mean()
    summary["equivalence-classes-histogram"] = [
        int(counts.get(c, 0)) for c in range(1, 11)
    ]
    summary["raw-quality-at-k"] = quality_at_k_arr
    summary["quality-at-k"] = quality_at_k_arr
    rbq = np.average(quality_at_k_arr, weights=decay_rate ** np.arange(10))
    summary["rank-biased-quality"] = rbq.item()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    df = pd.read_json(f"evals/{args.model}/scores.jsonl", lines=True)
    summary = summarize(df)
    with open(f"evals/{args.model}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
