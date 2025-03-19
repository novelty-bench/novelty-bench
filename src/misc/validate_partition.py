import glob
import json
import os

def validate(eval_dirs, eval_sets, models):
    for eval_dir in eval_dirs:
        for model in models:
            for eval_set in eval_sets:
                eval_file = f"{eval_dir}/{eval_set}/{model}/partitions.jsonl"
                print("validating", eval_file)
                
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
                        partition = data["partition"]
                        if len(partition) != 10:
                            error_count += 1
                    except json.JSONDecodeError:
                        print(f"Line {i} in {eval_file} is not valid JSON")

                if error_count > 0:
                    print(
                        f"{eval_file} has {error_count} lines with incorrect number of generations"
                    )



models = []
for model_type in ["ANTHROPIC", "OPENAI", "COHERE", "GEMINI", "VLLM"]:
    with open(f"model-lists/{model_type}_MODELS") as f:
        if model_type in ["ANTHROPIC", "COHERE", "GEMINI", "OPENAI"]:
            prefix = model_type.lower() + "/"
        else:
            prefix = ""
        models.extend(prefix + line.strip() for line in f if line.strip())

validate(["eval"], ["curated", "wildchat"], models)

models = []
for model_type in ["ANTHROPIC", "OPENAI"]:
    with open(f"model-lists/{model_type}_MODELS") as f:
        if model_type in ["ANTHROPIC", "COHERE", "GEMINI", "OPENAI"]:
            prefix = model_type.lower() + "/"
        else:
            prefix = ""
        models.extend(prefix + line.strip() for line in f if line.strip())
validate(["eval-ic", "eval-paraphrase", "eval-system-prompt"], ["curated"], models)


print("fi")