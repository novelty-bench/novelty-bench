import functools
from pathlib import Path

import tiktoken
from datasets import load_dataset


@functools.cache
def tokenizer():
    return tiktoken.encoding_for_model("gpt-4o")


def process_wildchat_instance(x: dict) -> dict:
    prompt = x["conversation"][0]["content"]
    id = f"WildChat-{x['conversation_hash']}"
    return {
        "id": id,
        "prompt": prompt,
    }


def filter_wildchat_instance(x: dict) -> bool:
    prompt = x["conversation"][0]["content"]
    return (
        x["language"] == "English"
        and (not x["redacted"])
        and 5 <= len(tokenizer().encode(prompt, disallowed_special=())) <= 200
    )


def main():
    dataset = load_dataset("allenai/WildChat-1M", split="train")
    subset = (
        dataset.filter(filter_wildchat_instance, num_proc=24)
        .map(process_wildchat_instance, num_proc=24, remove_columns=dataset.features)
        .shuffle(seed=1589180485)
    )

    subset_df = subset.to_pandas().drop_duplicates(["id", "prompt"])
    write_splits(subset_df)


def write_splits(subset_df, output_dir="data/wildchat"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_df.iloc[:5000].to_json(output_dir / "5k.jsonl", lines=True, orient="records")
    subset_df.iloc[0:5000].to_json(
        output_dir / "benchmark-no-labels.jsonl", lines=True, orient="records"
    )
    subset_df.iloc[5000:5100].to_json(
        output_dir / "dev-no-labels.jsonl", lines=True, orient="records"
    )
    subset_df.iloc[5100:5200].to_json(
        output_dir / "test-no-labels.jsonl", lines=True, orient="records"
    )


if __name__ == "__main__":
    main()
