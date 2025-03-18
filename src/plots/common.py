import pandas as pd
import altair as alt


@alt.theme.register('palatino', enable=True)
def palatino():
    font = "TeX Gyre Pagella"

    return {
        "config": {
            "title": {"font": font},
            "axis": {"labelFont": font, "titleFont": font},
            "header": {"labelFont": font, "titleFont": font},
            "legend": {"labelFont": font, "titleFont": font},
            "text": {"labelFont": font, "titleFont": font, "font": font},
        }
    }


models = [
    ("Anthropic", "anthropic/claude-3-5-haiku@20241022", "Claude 3.5 Haiku"),
    ("Anthropic", "anthropic/claude-3-5-sonnet-v2@20241022", "Claude 3.5 Sonnet"),
    ("Anthropic", "anthropic/claude-3-opus@20240229", "Claude 3 Opus"),
    
    ("OpenAI", "openai/gpt-4o-mini-2024-07-18", "GPT-4o Mini"),
    ("OpenAI", "openai/gpt-4o-2024-11-20", "GPT-4o"),
    
    ("Gemini", "gemini/gemini-1.5-pro", "Gemini 1.5 Pro"),
    ("Gemini", "gemini/gemini-2.0-flash-lite-preview-02-05", "Gemini 2.0 Flash Lite"),
    ("Gemini", "gemini/gemini-2.0-flash", "Gemini 2.0 Flash"),
    ("Gemini", "gemini/gemini-2.0-pro-exp-02-05", "Gemini 2.0 Pro"),
    
    ("Cohere", "cohere/command-r7b-12-2024", "Command R7B"),
    ("Cohere", "cohere/command-r-08-2024", "Command R"),
    ("Cohere", "cohere/command-r-plus-08-2024", "Command R+"),
    
    ("Gemma", "google/gemma-2-2b-it", "Gemma 2 2B"),
    ("Gemma", "google/gemma-2-9b-it", "Gemma 2 9B"),
    ("Gemma", "google/gemma-2-27b-it", "Gemma 2 27B"),
    ("Llama", "meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B"),
    ("Llama", "meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B"),
    ("Llama", "meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B"),
    ("Llama", "meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B"),
    ("Llama", "meta-llama/Llama-3.1-405B-Instruct", "Llama 3.1 405B")
]

model_order = [model_alias for _, _, model_alias in models]

score_dfs = []

for subset in ["wildchat", "curated"]:
    for model_family, model_path, model_alias in models:
        df = pd.read_json(f"eval/{subset}/{model_path}/scores.jsonl", lines=True)
        df["subset"] = subset
        df["model_family"] = model_family
        df["model_alias"] = model_alias
        score_dfs.append(df)

model_scores = pd.concat(score_dfs)
model_scores["distinct"] = model_scores["partition_scores"].map(len)
