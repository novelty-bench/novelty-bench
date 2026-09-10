# NoveltyBench

See [project webpage](https://novelty-bench.github.io/) for the dataset, evaluation results and instructions for submitting new models.

## Installation

Python 3.11 or newer.

via uv:
```shell
uv sync
```

via pip:
```shell
pip install -e .
```

Run every stage as a module — `python -m src.partition`, not
`python src/partition.py`. The scripts import each other through the `src`
package, and the module form resolves that against the directory you are
standing in; the script form resolves it against wherever the package happens
to be installed, which silently runs the wrong copy if you have more than one
checkout.

### API keys

Partitioning and scoring in v1.1 call hosted judges, and most inference
providers are hosted too. Export whichever you need:

```shell
export ANTHROPIC_API_KEY=...   # claude-* judges and inference
export OPENAI_API_KEY=...      # gpt-* judges and inference
```

`OPENAI_API_KEY` can instead live in a file named `openai-api-key` in the
repository root. Other providers read `cohere-api-key`, `gemini-api-key`,
`together-api-key` or `openrouter-api-key` from the root the same way; Vertex
providers use application-default credentials plus `--project` / `--region`.
Nothing in the pipeline needs a key until it makes its first call, so
`--limit 2` is a cheap way to check a provider is wired up.

## Usage

### Basic Workflow

1. **Inference**: Generate multiple responses from language models

   ```shell
   python -m src.inference --mode openai --model gpt-5.6-luna --data curated \
     --eval-dir results/curated/luna --num-generations 10 \
     --max-tokens 2048 --reasoning-effort low
   ```

2. **Partition**: Group responses that say the same thing

   ```shell
   python -m src.partition --eval-dir results/curated/luna --concurrency 32
   ```

3. **Score**: Judge one response per distinct answer

   ```shell
   python -m src.score --eval-dir results/curated/luna --concurrency 16 --patience 0.8
   ```

4. **Summarize**: Reduce to `mean_distinct` and `mean_utility`

   ```shell
   python -m src.summarize --eval-dir results/curated/luna
   ```

Steps 2-4 take `--version` (default `1.1`) and read/write
`<eval-dir>/v<version>/{partitions,scores}.jsonl` and `summary.json`, so one
`eval-dir` can hold results under several metric versions:

```
results/curated/luna/
  generations.jsonl        # inference output, shared by every version
  v1.0/                    # partitions.jsonl, scores.jsonl, summary.json
  v1.1/                    # partitions.jsonl, scores.jsonl, summary.json
```

Submissions made before v1.1 have their v1.0 files at the top level rather than
under `v1.0/`; both layouts are read.

### Resuming and caching

Every stage is resumable and safe to re-run. Each output row carries a
`*_key` hash of its inputs and a `*_config` record of the settings that
produced it (`generation_key`, `partition_key`, `score_key`), so a re-run
recomputes only the rows whose prompt, responses, partition or configuration
actually changed, and leaves the rest untouched. Change the judge model or the
patience and the affected stage recomputes; change nothing and it is a no-op.

While a stage runs, finished rows are appended to `<file>.partial`; the real
output file is replaced atomically only once every row is in. An interrupted
run therefore costs nothing — start the same command again and it picks up from
the journal. The journal is removed when the stage completes.

### Metric versions

| version | partition | utility |
|---|---|---|
| 1.0 | `classifier`: pairwise DeBERTa similarity classifier, each response compared to the head of each existing class (128-token window, see below) | Skywork-Reward-Gemma-2-27B reward model |
| 1.1 | `llm`: one `gpt-5.6-luna` call per prompt sees all responses in full and returns the groups (`JUDGE_SYSTEM` in `src/partition.py`) | `llm`: `claude-opus-5` scores the first response of each class on its own, 1-10, by how well it serves what the prompt asked for (`SCORE_SYSTEM` in `src/score.py`); duplicates are left unscored |

`--alg` / `--scorer` override a version's defaults and `--judge-model` the judge
(`claude-*` needs `ANTHROPIC_API_KEY`, `gpt-*` needs `OPENAI_API_KEY`); `--seed`
shuffles the order responses are shown to the partition judge, for stability
checks. Results are cached by content and config, and an interrupted run
resumes from `<file>.partial`.

**v1.1 in short.** The v1.0 partition classifier saw only the first 128 tokens
of each response and compared each response to one class head, so it merged
different stories and split near-identical ones; the v1.0 reward model scored
typicality rather than quality and read a focused but unusual answer as a bad
one. v1.1 replaces both with LLM judges whose prompts are the metric definition:
a response is *distinct* if a reader of another response would still gain
something from it, and its *utility* is how well it serves what the prompt asked
for — one thing asked for is best served by one thing, elaboration beyond the
ask earns nothing, and craft counts where the prompt calls for it. On a 60-instance
sample, judge partitions agreed with each other at ARI 0.9–0.97, against
0.3–0.8 for the classifier; re-partitioning every leaderboard model's curated
set with the responses shown in a different order reproduced the shipped
partition at mean ARI 0.89 (0.79–0.95), moving no model's distinct count by
more than 0.26; utility judges agreed at Spearman 0.83 (Opus vs Sol) and ~0.55 with
the reward model. The generation protocol also changed for new submissions:
`max_tokens` 2048 (v1.0 used 512, which truncated 20–55% of most models'
wildchat responses), reasoning effort low, and no sampling parameters on models
that reject them. v1.0 numbers remain on the leaderboard under their own tab.

For large runs, `src/batch.py` sends a stage through the Anthropic Batches API
at half price: `python src/batch.py submit --stage partition --eval-dir ...`,
then `status`, then `collect`, which writes the same files as a live run and
re-judges any invalid batch output live. Score after partitions are collected.

### Note on classifier input length

**Caution:** the `classifier` equivalence algorithm truncates **each response to
its first 128 tokens** before scoring (`max_length=128` in `src/partition.py`).
The same truncation was applied when the classifier was fine-tuned (`MAX_LEN` in
`src/classifier/finetune_classifier.py`), so this is the model's effective
comparison window rather than an inference-time mismatch — but it does mean that
any content past that window has no effect on the partition. Two responses that
share an opening and diverge only later will be scored as if they were identical.

This is worth checking before applying the classifier to long-form generations.
For reference, the released classifier training data (`data/train.jsonl`) has a
median response length of roughly 308 whitespace-delimited words, already well
beyond the 128-token window the model actually sees, so the window is a ceiling
on the comparison rather than a limit that long inputs merely approach. If your
responses carry their distinguishing content late — later turns, rebuttals,
conclusions — the partition may under-count distinct responses, and results are
best interpreted as a judgement about the opening of each response.

### Adding a model (v1.1 protocol)

Generation protocol for v1.1 submissions: 10 generations per prompt,
`--max-tokens 2048`, reasoning models at `--reasoning-effort low` with sampling
parameters unset, other models at temperature 1.0. Report which of two sampling
modes you used: `regenerate` (independent samples) or `in-context` (each sample
asked for in the same conversation, after the previous ones).

Note `--max-tokens` defaults to 512 in `src/inference.py`, the pre-v1.1 value,
so a v1.1 run has to pass it explicitly. `src/batch_generate.py` already
defaults to 2048.

1. Generate. `regenerate` runs can go through the provider's batch API at half
   price; `in-context` is sequential, so run it live (the Anthropic client puts a
   cache breakpoint on the latest turn, so the growing prefix is cached; OpenAI
   caches prefixes automatically).

   ```shell
   # regenerate, batched: submit, poll until it ends, then collect
   python -m src.batch_generate submit  --model claude-opus-5 --data curated --reasoning-effort low --eval-dir evaluation/<date>_claude-opus-5/nb-curated
   python -m src.batch_generate status  --model claude-opus-5 --data curated --reasoning-effort low --eval-dir evaluation/<date>_claude-opus-5/nb-curated
   python -m src.batch_generate collect --model claude-opus-5 --data curated --reasoning-effort low --eval-dir evaluation/<date>_claude-opus-5/nb-curated
   # in-context, live
   python -m src.inference --mode anthropic --model claude-opus-5 --data curated --sampling in-context --max-tokens 2048 --reasoning-effort low --concurrent-requests 8 --eval-dir evaluation/<date>_claude-opus-5_in-context/nb-curated
   ```
   `--mode openai` for OpenAI models (`gpt-5.6-*`, `gpt-6-*`); `anthropic` uses the
   Anthropic API directly, `anthropic-vertex` the Vertex endpoint. `submit`,
   `status` and `collect` all take the same flags, because the settings are part
   of the cache key. Add `--limit 2` to any of these for a smoke test.

2. Partition with the v1.1 judge (`gpt-5.6-luna`, set-level, live). `--eval-dir`
   accepts several directories, and a failure in one does not abandon the rest:

   ```shell
   python -m src.partition --version 1.1 --judge-model gpt-5.6-luna --concurrency 32 \
     --eval-dir evaluation/<date>_<model>/nb-curated evaluation/<date>_<model>/nb-wildchat
   ```

3. Score one response per distinct answer with `claude-opus-5`, then summarise.
   Live for a single split, or through the Batches API at half price for many:

   ```shell
   # live
   python -m src.score --version 1.1 --concurrency 16 --eval-dir evaluation/<date>_<model>/nb-curated
   # or batched: submit, poll, collect
   python -m src.batch submit  --stage score --eval-dir evaluation/<date>_<model>/nb-curated evaluation/<date>_<model>/nb-wildchat
   python -m src.batch status   --stage score --eval-dir evaluation/<date>_<model>/nb-curated evaluation/<date>_<model>/nb-wildchat
   python -m src.batch collect --stage score --eval-dir evaluation/<date>_<model>/nb-curated evaluation/<date>_<model>/nb-wildchat
   python -m src.summarize --version 1.1 --eval-dir evaluation/<date>_<model>/nb-curated
   ```

Submit `generations.jsonl` and `v1.1/{partitions,scores}.jsonl` plus
`v1.1/summary.json` for each of `nb-curated` and `nb-wildchat`.

### Batch or live

`src/batch.py` runs a judge stage and `src/batch_generate.py` runs generation
through the Anthropic and OpenAI batch APIs, at half the per-token price. Both
take `submit`, `status`, `collect`. `collect` writes exactly the files a live
run writes, through the same cache keys and the same atomic replace, and
re-judges any request that came back malformed, refused or expired using the
live API, so a batch run and a live run are interchangeable outputs.

The tradeoff is latency, not accuracy: a batch is promised within 24 hours and
usually lands sooner, but the queue is outside your control, and submitting many
large batches at once can leave them all pending for hours. Prefer batch for a
whole leaderboard, live for one model you want now. `collect` is idempotent —
re-run it until it reports every instance judged.

### Caveats worth knowing before a run

- **Claude output tokens include thinking.** `max_tokens` bounds visible text
  plus reasoning together, so a 2048-token visible cap needs a larger budget;
  the code requests double when `--reasoning-effort` is set.
- **A reasoning model can return no visible text**, having spent the whole
  budget thinking. That response is recorded as `[empty]` rather than retried
  forever, so the row keeps its 10 generations and the judge scores it as the
  failed answer it is. Count these before reading a model's utility.
- **A provider refusal is recorded as `[refused]`**, for the same reason.
- **A judge can decline to score a response.** `--fallback-judge` (default
  `claude-sonnet-5`) scores whatever the main judge declines, and those rows
  record `fallback_judge` and which calls it covered. It is error recovery
  rather than a metric setting, so it is deliberately not part of the cache key.
  If every judge declines, the row is marked `"unscored": "judge_refusal"`.
- **Pre-v1.1 submissions were generated at `max_tokens 512`**, which truncated
  20–55% of most models' WildChat responses mid-sentence. Their v1.0 and v1.1
  scores both measure those truncated responses; only new runs get 2048.
- **The v1.0 partition classifier needs a GPU** and downloads a DeBERTa
  checkpoint. The v1.1 judges need only API keys, so v1.1 reproduces on a laptop.

### Full Worked Example

For example, to run gemma-3-1b-it from start to finish:
```bash
export MODEL_NAME=google/gemma-3-1b-it
export SPLIT=curated
```

1. **Inference**: Generate multiple responses from language models
#### WITH VLLM
(may need to add model name to `model-lists/VLLM_MODELS` if not present)
Set up the VLLM server:
```bash
# Set environment variable for port
export VLLM_PORT=8000

# Start VLLM server
uv run vllm serve $MODEL_NAME --port 8000 --served-model-name $MODEL_NAME > vllm.log 2>&1 &
```

**Note**: The server takes 1-2 minutes to initialize and load the model.

```bash
uv run python -m src.inference \
  --mode vllm \
  --model $MODEL_NAME \
  --data $SPLIT \
  --eval-dir results/$SPLIT/$MODEL_NAME \
  --sampling regenerate \
  --num-generations 10
```
When done, kill the VLLM server:
```bash
pkill -f vllm
```

#### WITH TRANSFORMERS (slower than VLLM, but more flexible)

```bash
uv run python -m src.inference \
  --mode transformers \
  --model $MODEL_NAME \
  --data $SPLIT \
  --eval-dir results/$SPLIT/$MODEL_NAME \
  --sampling regenerate \
  --num-generations 10
```

Local models take no sampling-parameter or reasoning flags, so add
`--max-tokens 2048` to match the v1.1 protocol.

2. **Partition**: Group responses that say the same thing

```bash
uv run python -m src.partition \
  --eval-dir results/$SPLIT/$MODEL_NAME \
  --concurrency 32
```

3. **Score**: Judge one response per distinct answer

```bash
uv run python -m src.score \
  --eval-dir results/$SPLIT/$MODEL_NAME \
  --concurrency 16 \
  --patience 0.8
```

4. **Summarize**: Reduce to `mean_distinct` and `mean_utility`
```bash
uv run python -m src.summarize --eval-dir results/$SPLIT/$MODEL_NAME
```

To reproduce the original metric instead, pass `--version 1.0` to steps 2-4.
That selects the DeBERTa classifier and the Skywork reward model, needs a GPU,
and writes to `v1.0/` rather than `v1.1/`.


## Project Structure

- `src/`: Core source code
  - `inference.py`: Handles generation from various LLM providers
  - `batch_generate.py`, `batch.py`: Generation and judge stages through the OpenAI / Anthropic batch APIs
  - `judge.py`: Structured-output judge calls shared by live and batch paths
  - `partition.py`: Implements response partitioning algorithms
  - `score.py`: Scores one response per distinct answer (LLM judge in v1.1, reward model in v1.0)
  - `summarize.py`: Summarize evaluation results
  - `evaluation_io.py`: Cache keys, crash journals and atomic output replacement
  - `common.py`: Metric versions, directory layout and provider clients
- `data/`: Contains curated and wildchat datasets, human annotations, and classifier training data
- `evaluation/`: Contains evaluation results for leaderboard participation. We have provided an example submission.
- `tests/`: Offline regression tests, no network or API keys required:
  ```shell
  python -m unittest discover -s tests
  ```

## Where the data lives

Git keeps the metric; the dataset repo keeps the corpus. A `scores.jsonl` row
carries both the ten responses and the numbers computed from them, and the
responses are ~97% of the bytes — so the committed copies have the
`generations` field dropped (`python -m src.slim`) and every other field
intact. `python -m src.summarize` still reproduces a `summary.json` from them,
and a diff on a judged file shows a changed number rather than a wall of prose.

| where | what |
|---|---|
| this repo | `summary.json`, slim `partitions.jsonl` and `scores.jsonl`, and the curated split's `generations.jsonl` — 100 prompts of real response text, so a checkout is self-contained |
| [`yimingzhang/novelty-bench`](https://huggingface.co/datasets/yimingzhang/novelty-bench) | every run's full-fidelity files under `results/<run>/nb-<split>/`, response text included |
| nowhere | `batch.*` working files from the Batches API |

```shell
# fetch a run's full files, or one of them
python -m src.publish pull --run 2026-09-10_claude-opus-5 --eval-dir /tmp/opus-5
python -m src.publish pull --run 2026-09-10_claude-opus-5 --eval-dir /tmp/opus-5 \
  --file nb-wildchat/generations.jsonl

# publish your own run (needs a write token: huggingface-cli login)
python -m src.publish push --eval-dir evaluation/<date + name> --repo-id <you>/<dataset>
```

## 🏆 Leaderboard Participation

If you are interested in submitting your model to the NoveltyBench Leaderboard, please do the following:

1. Fork this repository;
2. Clone your fork;
3. Under `evaluation/`, create a new folder with the submission date and your model name (e.g., `2025-03-27_gemini-1.5-pro`);
4. Within the folder (`evaluation/<date + name>/`), commit these files for each
   subset _NB-Curated_ and _NB-WildChat_:
    ```
    - generations.jsonl        (curated only; see Where the data lives)
    - v1.1/partitions.jsonl    slim
    - v1.1/scores.jsonl        slim
    - v1.1/summary.json
    ```
   `evaluation/2026-09-10_claude-opus-5/README.md` is the format reference: it
   names every file and field. Slim the two judged files before committing, and
   push the full-fidelity run to a dataset repo:
    ```shell
    python -m src.slim evaluation/<date + name>/nb-*/v1.1/*.jsonl
    python -m src.publish push --eval-dir evaluation/<date + name>
    ```
  - Generate with the v1.1 protocol described in
    [Adding a model](#adding-a-model-v11-protocol): 10 responses per prompt at
    `--max-tokens 2048`, reasoning models at `--reasoning-effort low`. Say in
    the pull request which sampling mode you used, `regenerate` or
    `in-context`, and note any `[empty]` or `[refused]` placeholders.
5. Create a pull request to this repository with the new folder.
6. (Optional) To get attribution on the leaderboard, include in your PR description:
   ```json
   {
     "paper": "https://arxiv.org/abs/...",
     "model": "https://huggingface.co/...",
     "authors": "LastName et al."
   }
   ```

The NoveltyBench team will:
- Review and merge your submission;
- Update the leaderboard with your results.

### Submission policy

The leaderboard is for training and inference-time methods applied to publicly
available models (open weights, or a public API with a named model version).
Closed systems and results we can't reproduce are out of scope and will be
closed without review. Anyone can run NoveltyBench on any system and publish
the results, citing the benchmark. Not sure if your method is in scope? Open an
issue first.


## Contact

If you have any questions, please create an issue. Otherwise, you can also contact us via email at `yimingz3@cs.cmu.edu`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
