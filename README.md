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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
