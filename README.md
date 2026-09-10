# NoveltyBench

See [project webpage](https://novelty-bench.github.io/) for the dataset, evaluation results and instructions for submitting new models.

## Installation

via pip:
```shell
# Install dependencies
pip install -e .
```

via uv:
```shell
uv sync
```

## Usage

### Basic Workflow

1. **Inference**: Generate multiple responses from language models

   ```shell
   python src/inference.py --mode openai --model gpt-4o --data curated --eval-dir results/curated/gpt4o --num-generations 10
   ```

2. **Partition**: Group semantically similar responses

   ```shell
   python src/partition.py --eval-dir results/curated/gpt4o --concurrency 32
   ```

3. **Score**: Evaluate the quality of responses

   ```shell
   python src/score.py --eval-dir results/curated/gpt4o --patience 0.8
   ```

4. **Summarize**: Analyze and visualize results

   ```shell
   python src/summarize.py --eval-dir results/curated/gpt4o
   ```

Steps 2-4 take `--version` (default `1.1`) and read/write
`<eval-dir>/v<version>/{partitions,scores}.jsonl` and `summary.json`, so one
`eval-dir` can hold results under several metric versions.

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
sample, judge partitions agreed with each other at ARI 0.9–0.97 and were
order-stable (ARI 0.92–0.95 across shuffles), against 0.3–0.8 for the
classifier; utility judges agreed at Spearman 0.83 (Opus vs Sol) and ~0.55 with
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
parameters unset, other models at temperature 1.0. Reasoning models get a
4096-token budget shared with their reasoning; a response whose budget ran out
before any visible text is recorded as `[empty]`, and a provider refusal as
`[refused]`, so every row keeps 10 generations. Submissions made before v1.1 were generated at `max_tokens 512`,
which truncated 20–55% of responses for most models. Two sampling modes are
reported: `regenerate` (independent samples) and `in-context` (each sample is
asked for in the same conversation after the previous ones).

1. Generate. `regenerate` runs can go through the provider's batch API at half
   price; `in-context` is sequential, so run it live (the Anthropic client puts a
   cache breakpoint on the latest turn, so the growing prefix is cached; OpenAI
   caches prefixes automatically).

   ```shell
   # regenerate, batched: submit, then poll status, then collect
   python src/batch_generate.py submit  --model claude-opus-5 --data curated --reasoning-effort low --eval-dir evaluation/<date>_claude-opus-5/nb-curated
   python src/batch_generate.py collect --model claude-opus-5 --data curated --reasoning-effort low --eval-dir evaluation/<date>_claude-opus-5/nb-curated
   # in-context, live
   python src/inference.py --mode anthropic --model claude-opus-5 --data curated --sampling in-context --max-tokens 2048 --reasoning-effort low --concurrent-requests 8 --eval-dir evaluation/<date>_claude-opus-5_in-context/nb-curated
   ```
   `--mode openai` for OpenAI models (`gpt-5.6-*`, `gpt-6-*`); `anthropic` uses the
   Anthropic API directly, `anthropic-vertex` the Vertex endpoint. Rows carry
   `generation_config`, so a live run and a batch collect recognise each other's
   output and an interrupted run resumes.

2. Partition with the v1.1 judge (`gpt-5.6-luna`, set-level, live):

   ```shell
   python src/partition.py --version 1.1 --judge-model gpt-5.6-luna --concurrency 32 --eval-dir evaluation/<date>_<model>/nb-*
   ```

3. Score class heads with `claude-opus-5` through the Batches API, then summarise:

   ```shell
   python src/batch.py submit  --stage score --eval-dir evaluation/<date>_<model>/nb-*
   python src/batch.py collect --stage score --eval-dir evaluation/<date>_<model>/nb-*
   python src/summarize.py --version 1.1 --eval-dir evaluation/<date>_<model>/nb-curated
   ```

Repeat for `nb-wildchat`. Submit `generations.jsonl` and `v1.1/{partitions,scores}.jsonl`
plus `v1.1/summary.json` per split.

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
uv run python src/inference.py \
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
uv run python src/inference.py \
  --mode transformers \
  --model $MODEL_NAME \
  --data $SPLIT \
  --eval-dir results/$SPLIT/$MODEL_NAME \
  --sampling regenerate \
  --num-generations 10
```

2. **Partition**: Group semantically similar responses

```bash
uv run python src/partition.py \
  --eval-dir results/$SPLIT/$MODEL_NAME \
  --alg classifier
```

3. **Score**: Evaluate the quality of responses

```bash
uv run python src/score.py \
  --eval-dir results/$SPLIT/$MODEL_NAME \
  --patience 0.8
```

4. **Summarize**: Analyze and visualize results
```bash
uv run python src/summarize.py --eval-dir results/$SPLIT/$MODEL_NAME
```


## Project Structure

- `src/`: Core source code
  - `inference.py`: Handles generation from various LLM providers
  - `batch_generate.py`, `batch.py`: Generation and judge stages through the OpenAI / Anthropic batch APIs
  - `judge.py`: Structured-output judge calls shared by live and batch paths
  - `partition.py`: Implements response partitioning algorithms
  - `score.py`: Scores class heads (LLM judge in v1.1, reward model in v1.0)
  - `summarize.py`: Summarize evaluation results
- `data/`: Contains curated and wildchat datasets, human annotations, and classifier training data
- `evaluation/`: Contains evaluation results for leaderboard participation. We have provided an example submission.

## 🏆 Leaderboard Participation

If you are interested in submitting your model to the NoveltyBench Leaderboard, please do the following:

1. Fork this repository;
2. Clone your fork;
3. Under `evaluation/`, create a new folder with the submission date and your model name (e.g., `2025-03-27_gemini-1.5-pro`);
4. Within the folder (`evaluation/<date + name>/`), please include the following **required** assets:
  - Follow the instruction in the Basic Workflow section to get the following files for each subset _NB-Curated_ and _NB-WildChat_:
    ```
    - generations.jsonl
    - v1.1/partitions.jsonl
    - v1.1/scores.jsonl
    - v1.1/summary.json
    ```
  - Put your **scores.jsonl** and **summary.json** under the folder. You final folder should look like:
    ```
    - evaluation/
      - <date + name>/
        - nb-curated/
          - v1.1/
            - scores.jsonl
            - summary.json
        - nb-wildchat/
          - v1.1/
            - scores.jsonl
            - summary.json
    ```
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
