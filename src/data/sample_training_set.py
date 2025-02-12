import glob
import json
import random
from pathlib import Path
import math
import pandas as pd


def find_generation_files(path) -> list[str]:
    eval_dir = Path(path)
    all_files = glob.glob(str(eval_dir / "**" / "generations.jsonl"), recursive=True)
    # Filter out files containing "DeepSeek"
    return [f for f in all_files if "DeepSeek" not in f]


def sample_random_generations(generations):
    ...


def main():
    files = []
    for eval_root in ["eval", "eval-ic"]:
        for subset in ["curated", "wildchat"]:
            files = files + find_generation_files(path=f"{eval_root}/{subset}")
    print(f"Found {len(files)} generation files")
    data = pd.concat(pd.read_json(f, lines=True) for f in files)

    data = data[data["generations"].map(len) == 10]
    data = data.sample(frac=1.0).drop_duplicates(["id"])
    print(len(data))

    samples = []
    for _, row in data.iterrows():
        row = row.to_dict()
        i, j = random.sample(range(10), 2)
        row["generation_0"], row["generation_1"] = (
            row["generations"][i],
            row["generations"][j],
        )
        samples.append(row)
    
    
    samples_df = pd.DataFrame(samples)
    labelers = ["Barry", "Daphne", "Diddee", "Kevin", "Sue", "Vinay", "Xinyue", "Yiming"]
    samples_df["labeler"] = samples_df.index.map(
        lambda i: labelers[math.floor(i / len(samples) * len(labelers))]
    )
    samples_df.to_json("data/classifier/unlabeled.jsonl", lines=True, orient="records")
    samples_df.to_csv("data/classifier/unlabeled.csv")

if __name__ == "__main__":
    main()
