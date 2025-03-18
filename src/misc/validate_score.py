import glob
import json
import os

models = []
for model_file in glob.glob("model-lists/*"):
    with open(model_file) as f:
        prefix = ""
        for model_type in ["ANTHROPIC", "COHERE", "GEMINI", "OPENAI", "VERTEX"]:
            if model_type in model_file:
                prefix = model_type.lower() + "/"
        models.extend(prefix + line.strip() for line in f if line.strip())


for eval_dir in ["eval", "eval-ic"]:
    for model in models:
        for eval_set in ["curated", "wildchat"]:
            eval_file = f"{eval_dir}/{eval_set}/{model}/scores.jsonl"
            expected_lines = 100 if eval_set == "curated" else 1000

            if not os.path.exists(eval_file):
                print(f"Error: {eval_file} does not exist")
                continue

            with open(eval_file) as f:
                lines = [line.strip() for line in f if line.strip()]

            if len(lines) != expected_lines:
                print(
                    f"{eval_file} should have {expected_lines} lines, but has {len(lines)} lines"
                )

            error_count = 0
            for i, line in enumerate(lines, 1):
                try:
                    data = json.loads(line)
                    scores = data["partition_scores"]
                    if len(scores) != max(data["partition"]) + 1:
                        error_count += 1
                    elif len(data["generation_scores"]) != 10:
                        error_count += 1
                except json.JSONDecodeError:
                    print(f"Line {i} in {eval_file} is not valid JSON")

            if error_count > 0:
                print(
                    f"{eval_file} has {error_count} lines with incorrect number of generations"
                )

print("fi")