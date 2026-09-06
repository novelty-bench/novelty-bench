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
   python src/partition.py --eval-dir results/curated/gpt4o --alg classifier
   ```

3. **Score**: Evaluate the quality of responses

   ```shell
   python src/score.py --eval-dir results/curated/gpt4o --patience 0.8
   ```

4. **Summarize**: Analyze and visualize results

   ```shell
   python src/summarize.py --eval-dir results/curated/gpt4o
   ```

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
  - Follow the instruction in the Basic Workflow section to get the following files for each subset _NB-Curated_ and _NB-WildChat_:
    ```
    - generations.jsonl
    - partitions.jsonl
    - scores.jsonl
    - summary.json
    ```
  - Put your **scores.jsonl** and **summary.json** under the folder. You final folder should look like:
    ```
    - evaluation/
      - <date + name>/
        - nb-curated/
          - scores.jsonl
          - summary.json
        - nb-wildchat/
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

