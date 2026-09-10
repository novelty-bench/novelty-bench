import argparse
import json
import os

import numpy as np
import pandas as pd

from src.common import DEFAULT_VERSION, METRIC_VERSIONS, version_dir


def summarize(df: pd.DataFrame, version: str) -> dict:
    summary = {"version": version}

    # An instance the judge declined carries no scores; it is left out of the
    # utility mean rather than counted as a bad answer, and reported instead.
    scored = df[df["utility"].notna()] if "utility" in df else df
    summary["mean_distinct"] = np.mean(scored["partition_scores"].map(len))
    summary["mean_utility"] = np.mean(scored["utility"])
    if len(scored) < len(df):
        summary["unscored"] = int(len(df) - len(scored))
        summary["n_scored"] = int(len(scored))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    parser.add_argument("--version", default=DEFAULT_VERSION, choices=METRIC_VERSIONS)
    args = parser.parse_args()

    eval_dir = version_dir(args.eval_dir, args.version)
    df = pd.read_json(os.path.join(eval_dir, "scores.jsonl"), lines=True)
    summary = summarize(df, args.version)
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
