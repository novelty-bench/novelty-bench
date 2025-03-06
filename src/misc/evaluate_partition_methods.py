import argparse
import asyncio
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm.auto import tqdm

from src.partition import (
    equivalence_check_gpt4,
    equivalence_check_lcs,
    equivalence_check_bertscore,
)


async def evaluate_method(df, method_name, equivalence_check_fn):
    """
    Evaluate a partitioning method on the labeled dataset.
    
    Args:
        df: DataFrame containing the labeled dataset
        method_name: Name of the method being evaluated
        equivalence_check_fn: Function to check equivalence between two responses
        
    Returns:
        Dictionary containing precision, recall, F1, and accuracy
    """
    print(f"\nEvaluating {method_name}...")
    
    # Ground truth labels (1 for "Same", 0 for "Different")
    y_true = df["binary_label"].tolist()
    
    # Predicted labels
    y_pred = []
    
    # Process each row
    for i, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row["prompt"]
        gen_0 = row["generation_0"]
        gen_1 = row["generation_1"]
        
        # Check equivalence
        is_equivalent = await equivalence_check_fn(prompt, gen_0, gen_1)
        
        # Convert to binary label (1 for "Same", 0 for "Different")
        y_pred.append(1 if is_equivalent else 0)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    tp = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    fp = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    fn = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))
    tn = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    
    # Print results
    print(f"\n{method_name} Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives: {tn}")
    
    return {
        "method": method_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


async def main():
    parser = argparse.ArgumentParser(description="Evaluate partitioning methods")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/classifier/labeled.csv",
        help="Path to the labeled data CSV file",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gpt4", "lcs", "bertscore"],
        choices=["gpt4", "lcs", "bertscore"],
        help="Methods to evaluate",
    )
    args = parser.parse_args()
    
    # Read the CSV file
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    
    # Clean the data
    df = df.dropna(subset=["label (Same / Different)"])
    
    # Convert labels from "Same" / "Different" to 1 / 0
    df["binary_label"] = df["label (Same / Different)"].apply(
        lambda x: 1 if x.strip().upper() in ["S", "SAME"] else (0 if x.strip().upper() in ["D", "DIFFERENT"] else None)
    )
    
    # Drop rows with invalid labels
    df = df.dropna(subset=["binary_label"])
    def truncation(s):
        return '\n'.join(s.split('\n')[:10])
    df["generation_0"] = df["generation_0"].map(truncation)
    df["generation_1"] = df["generation_1"].map(truncation)
    
    # Print label distribution
    label_counts = df["binary_label"].value_counts()
    total = len(df)
    print("\nLabel Distribution:")
    print(f"Same (1): {label_counts.get(1, 0)} ({(label_counts.get(1, 0)/total)*100:.1f}%)")
    print(f"Different (0): {label_counts.get(0, 0)} ({(label_counts.get(0, 0)/total)*100:.1f}%)")
    print(f"Total samples: {total}\n")
    
    # Map method names to functions
    method_map = {
        "gpt4": equivalence_check_gpt4,
        "lcs": equivalence_check_lcs,
        "bertscore": equivalence_check_bertscore,
    }
    
    # Evaluate each method
    results = []
    for method_name in args.methods:
        result = await evaluate_method(df, method_name, method_map[method_name])
        results.append(result)
    
    # Print comparison table
    print("\nComparison of Methods:")
    print(f"{'Method':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Accuracy':<10}")
    print("-" * 50)
    for result in results:
        print(f"{result['method']:<10} {result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1']:<10.4f} {result['accuracy']:<10.4f}")


if __name__ == "__main__":
    asyncio.run(main()) 