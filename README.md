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

Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`; the v1.1 judges call both.

1. **Inference**: Generate multiple responses from language models

   ```shell
   python -m src.inference --mode openai --model gpt-5.6-luna --data curated \
     --eval-dir results/curated/luna --num-generations 10 \
     --max-tokens 2048 --reasoning-effort low
   ```

2. **Partition**: Group semantically similar responses

   ```shell
   python -m src.partition --eval-dir results/curated/luna --concurrency 32
   ```

3. **Score**: Evaluate the quality of responses

   ```shell
   python -m src.score --eval-dir results/curated/luna --concurrency 16 --patience 0.8
   ```

4. **Summarize**: Analyze and visualize results

   ```shell
   python -m src.summarize --eval-dir results/curated/luna
   ```

Steps 2-4 take `--version` and default to `1.1`, reading and writing
`<eval-dir>/v1.1/`. Pass `--version 1.0` for the original classifier and reward
model. Every stage is resumable: rows are cached by a hash of their inputs and
settings, and journalled to `<file>.partial` while a run is in flight.

### Metric versions

| version | partition | utility |
|---|---|---|
| 1.0 | DeBERTa similarity classifier, first 128 tokens of each response | Skywork-Reward-Gemma-2-27B reward model |
| 1.1 | one `gpt-5.6-luna` call per prompt reads all ten responses in full | `claude-opus-5` scores the first response of each group, 1-10 |

The v1.1 judge prompts are `JUDGE_SYSTEM` in `src/partition.py` and
`SCORE_SYSTEM` in `src/score.py`, and they are the metric definition. Judges
agree on partitions at ARI 0.90-0.97 against 0.29-0.80 for the classifier, and
reproduce a partition at mean ARI 0.89 when the responses are shown in a
different order. The two versions are not comparable; both are kept.

New submissions generate with `--max-tokens 2048` and `--reasoning-effort low`.
v1.0 submissions used 512, which truncated 20-55% of most models' WildChat
responses. On Claude models `max_tokens` also bounds thinking, so the code asks
for double; a response that spends its whole budget thinking is recorded as
`[empty]` and a provider refusal as `[refused]`.

For a whole leaderboard, `src/batch.py` (judging) and `src/batch_generate.py`
(generation) run a stage through the providers' batch APIs at half price:
`submit`, then `status`, then `collect`, which writes exactly what a live run
writes.

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

2. **Partition**: Group semantically similar responses

```bash
uv run python -m src.partition --eval-dir results/$SPLIT/$MODEL_NAME
```

3. **Score**: Evaluate the quality of responses

```bash
uv run python -m src.score \
  --eval-dir results/$SPLIT/$MODEL_NAME \
  --patience 0.8
```

4. **Summarize**: Analyze and visualize results
```bash
uv run python -m src.summarize --eval-dir results/$SPLIT/$MODEL_NAME
```


## Project Structure

- `src/`: Core source code
  - `inference.py`: Handles generation from various LLM providers
  - `partition.py`: Implements response partitioning algorithms
  - `score.py`: Computes utility scores using reward model
  - `summarize.py`: Summarize evaluation results
- `data/`: Contains curated and wildchat datasets, human annotations, and classifier training data
- `evaluation/`: Contains evaluation results for leaderboard participation. We have provided an example submission.

## 🏆 Leaderboard Participation

If you are interested in submitting your model to the NoveltyBench Leaderboard, please do the following:

1. Fork this repository;
2. Clone your fork;
3. Under `evaluation/`, create a new folder with the submission date and your model name (e.g., `2025-03-27_gemini-1.5-pro`);
4. Within the folder (`evaluation/<date + name>/`), please include the following **required** assets:
  - Follow the instruction in the Basic Workflow section to get, for each subset _NB-Curated_ and _NB-WildChat_, a `v1.1/` directory holding `partitions.jsonl`, `scores.jsonl` and `summary.json`.
  - Run `python -m src.slim` over the two jsonl files first: it drops the response text, which is ~97% of the bytes and none of the metric. Your folder should look like:
    ```
    - evaluation/
      - <date + name>/
        - nb-curated/
          - v1.1/
            - partitions.jsonl
            - scores.jsonl
            - summary.json
        - nb-wildchat/
          - v1.1/ ...
    ```
  - Publish the full files, response text included, with `python -m src.publish push --eval-dir <folder> --repo-id <you>/<dataset>`, and link the dataset in your PR. `python -m src.publish pull` fetches a published run back. `evaluation/2026-09-10_claude-opus-5/` is a worked example of the whole layout.
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
