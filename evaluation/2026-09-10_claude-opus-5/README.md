# Submission format

This run is the format reference: `claude-opus-5`, ten independent samples per
prompt, judged under metric v1.1. Copy this tree's shape for a new submission.

```
2026-09-10_claude-opus-5/
  README.md
  nb-curated/                   100 prompts
    generations.jsonl           committed: the responses themselves
    v1.1/
      partitions.jsonl          committed, slim
      scores.jsonl              committed, slim
      summary.json              committed: the two leaderboard numbers
  nb-wildchat/                  1,000 prompts, same files
```

One JSON object per line, one line per prompt, in both splits.

| file | fields on each row |
|---|---|
| `generations.jsonl` | `id`, `prompt`, `model`, `generations` (ten strings), `generation_key`, `generation_config` |
| `partitions.jsonl` | the above plus `partition` (ten class labels, `0`-based, in order of first appearance) and `distinct`, plus `partition_key`, `partition_config` |
| `scores.jsonl` | the above plus `generation_scores` (ten entries; `null` where a response repeated an earlier one and so was not judged), `partition_scores` (one per class), `utility`, plus `score_key`, `score_config`, and `unscored` if a judge declined the prompt |
| `summary.json` | `version`, `mean_distinct`, `mean_utility`, plus `unscored`/`n_scored` when a prompt was excluded |

## What is committed and what is not

`generations` is most of the bytes and none of the metric, so git keeps it only
for the 100 curated prompts. The WildChat response text, and the full
`partitions.jsonl` / `scores.jsonl` that embed it, live in the dataset repo:

```shell
# fetch this run's full-fidelity files
python -m src.publish pull --run 2026-09-10_claude-opus-5 --eval-dir /tmp/opus-5

# just the one file
python -m src.publish pull --run 2026-09-10_claude-opus-5 --eval-dir /tmp/opus-5 \
  --file nb-wildchat/generations.jsonl
```

The committed files are the same rows with `generations` dropped
(`python -m src.slim`), so every number here stays verifiable and diffable:
`python -m src.summarize --version 1.1 --eval-dir nb-curated` reproduces
`summary.json` from `scores.jsonl` alone.

`batch.*.json` and `batch.*.jsonl` are working files from the Batches API. They
are ignored by git and not published.

## How this run was produced

```shell
python -m src.inference --mode anthropic --model claude-opus-5 --data curated \
  --sampling regenerate --max-tokens 2048 --reasoning-effort low \
  --eval-dir evaluation/2026-09-10_claude-opus-5/nb-curated
python -m src.partition --version 1.1 --eval-dir <that dir> --concurrency 32
python -m src.score     --version 1.1 --eval-dir <that dir> --concurrency 16
python -m src.summarize --version 1.1 --eval-dir <that dir>
```

The companion run `2026-09-10_claude-opus-5_in-context` is the same model under
`--sampling in-context`, where each sample is requested in the same
conversation after the previous ones.
