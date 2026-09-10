"""Drop response text from an evaluation file, keeping every metric field.

`generations` is most of the bytes in a partitions or scores file and none of
the metric: the partition, the per-response scores and the utility are all
computed already. Git keeps the slim form so the numbers stay verifiable and
diffable; the full form, response text included, lives in the dataset repo
(see `src/publish.py`).
"""

import argparse
import json
import os

from src.evaluation_io import write_rows

DROP = ("generations",)


def slim_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in DROP}


def slim_file(path: str, out_path: str | None = None) -> int:
    """Rewrite `path` (or write `out_path`) without response text; returns rows."""
    with open(path) as f:
        rows = [slim_row(json.loads(line)) for line in f if line.strip()]
    write_rows(out_path or path, rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="+", help="partitions.jsonl or scores.jsonl")
    parser.add_argument("--out", help="write here instead of in place (single input)")
    args = parser.parse_args()
    if args.out and len(args.path) > 1:
        parser.error("--out takes a single input file")

    for path in args.path:
        before = os.path.getsize(path)
        rows = slim_file(path, args.out)
        after = os.path.getsize(args.out or path)
        print(f"{path}: {rows} rows, {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
