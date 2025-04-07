# NoveltyBench

See [project webpage](https://novelty-bench.github.io/) for the dataset, evaluation results and instructions for submitting new models.

## Installation

```shell
# Install dependencies
pip install -e .
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

## Project Structure

- `src/`: Core source code
  - `inference.py`: Handles generation from various LLM providers
  - `partition.py`: Implements response partitioning algorithms
  - `score.py`: Computes utility scores using reward model
  - `summarize.py`: Summarize evaluation results
- `data/`: Contains curated and wildchat datasets
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

The NoveltyBench team will:
- Review and merge your submission;
- Update the leaderboard with your results.


## Contact

If you have any questions, please create an issue. Otherwise, you can also contact us via email at `yimingz3@cs.cmu.edu`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
