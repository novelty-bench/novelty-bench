import argparse
import json
import os

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    summary = {}

    summary["mean_distinct"] = np.mean(df["partition_scores"].map(len))
    summary["mean_utility"] = np.mean(df["utility"])

    return summary


def main():
    parser = argparse.ArgumentParser()
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
