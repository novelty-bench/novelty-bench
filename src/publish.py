"""Move a run's full-fidelity results between an eval dir and a dataset repo.

Git keeps the slim files (see `src/slim.py`); the response text lives in a
HuggingFace dataset under `results/<run>/nb-<split>/`, mirroring the eval-dir
layout so a path maps to a repo path by prefix alone.

    python -m src.publish push --eval-dir evaluation/2026-09-10_my-model
    python -m src.publish pull --run 2026-09-10_my-model --eval-dir /tmp/my-model

Needs a write token for `push` (`huggingface-cli login`); `pull` works
anonymously against a public dataset.
"""

import argparse
import os

DEFAULT_REPO = "yimingzhang/novelty-bench"
PREFIX = "results"
# What is worth keeping: inputs, both judged stages, and the summary. Batch
# manifests and raw batch output are working files, not results.
PATTERNS = [
    "*/generations.jsonl",
    "*/v*/partitions.jsonl",
    "*/v*/scores.jsonl",
    "*/v*/summary.json",
]


def repo_path(run: str, *parts: str) -> str:
    """Where a run's file lives in the dataset repo."""
    return "/".join([PREFIX, run, *parts])


def run_name(eval_dir: str) -> str:
    return os.path.basename(os.path.normpath(eval_dir))


def push(api, eval_dir: str, repo_id: str, run: str | None = None) -> str:
    name = run or run_name(eval_dir)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=eval_dir,
        path_in_repo=repo_path(name),
        allow_patterns=PATTERNS,
        commit_message=f"add {name} evaluation results",
    )
    return repo_path(name)


def pull(api, run: str, eval_dir: str, repo_id: str, files=None) -> list[str]:
    """Fetch a run (or named files within it) into `eval_dir`.

    Downloads go through the hub cache and are copied out of it, so a rerun
    costs an ETag check per file rather than the bytes again.
    """
    import shutil

    from huggingface_hub import hf_hub_download

    if files:
        wanted = [repo_path(run, f) for f in files]
    else:
        wanted = [
            f
            for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            if f.startswith(repo_path(run) + "/")
        ]
    if not wanted:
        raise SystemExit(f"nothing under {repo_path(run)} in {repo_id}")

    written = []
    for remote in wanted:
        local = os.path.join(eval_dir, os.path.relpath(remote, repo_path(run)))
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        cached = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=remote)
        shutil.copyfile(cached, local)
        written.append(local)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--run", help="run name in the dataset (default: eval-dir name)")
    parser.add_argument(
        "--file",
        action="append",
        help="pull only this path within the run, e.g. nb-curated/generations.jsonl",
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    if args.action == "push":
        print(
            f"pushed to {args.repo_id}:{push(api, args.eval_dir, args.repo_id, args.run)}"
        )
    else:
        run = args.run or run_name(args.eval_dir)
        written = pull(api, run, args.eval_dir, args.repo_id, args.file)
        print(f"pulled {len(written)} files into {args.eval_dir}")


if __name__ == "__main__":
    main()
