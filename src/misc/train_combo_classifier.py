import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.partition import bleu, rouge1, bertscore
import json

def extract_features(df):
    """Extracts features (bleu, rouge1, bertscore) from the DataFrame."""
    features = []
    for _, row in df.iterrows():
        gen_0 = row["generation_0"]
        gen_1 = row["generation_1"]
        features.append([bleu(gen_0, gen_1), rouge1(gen_0, gen_1)
                         , bertscore(gen_0, gen_1)
                         ])
    return np.array(features)

def train_and_evaluate(train_file, val_file):
    """
    Trains a logistic regression classifier using bleu, rouge1, and bertscore features.
    Evaluates the classifier on the validation set.
    """

    # Load the training and validation data
    train_df = pd.read_json(train_file, lines=True)
    val_df = pd.read_json(val_file, lines=True)

    # Extract features from the training and validation sets
    X_train = extract_features(train_df)
    X_val = extract_features(val_df)

    # Print feature correlations
    feature_df = pd.DataFrame(X_train, columns=['BLEU', 'ROUGE-1', 'BERTScore'])
    print("Feature Correlations:")
    print(feature_df.corr())
    
    # Extract labels from the training and validation sets
    y_train = train_df["similar"]
    y_val = val_df["similar"]
    
    # Check class balance
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Validation class distribution: {np.bincount(y_val)}")

    # Train the logistic regression classifier with regularization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced')
    model = model.fit(X_train_scaled, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val_scaled)

    # Evaluate the classifier
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Print the results
    print("Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Print the actual model coefficients
    print("Feature weights:")
    for feature, coef in zip(['BLEU', 'ROUGE-1', 'BERTScore'], model.coef_[0]):
        print(f"  {feature}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")
    
    # Evaluate individual metrics for comparison
    print("\nIndividual metric performance:")
    for i, metric_name in enumerate(['BLEU', 'ROUGE-1', 'BERTScore']):
        single_feature_model = LogisticRegression(class_weight='balanced')
        single_feature_model.fit(X_train_scaled[:, i:i+1], y_train)
        y_pred_single = single_feature_model.predict(X_val_scaled[:, i:i+1])
        single_f1 = f1_score(y_val, y_pred_single)
        print(f"  {metric_name} F1: {single_f1:.4f}")

def main():
    train_file = "data/classifier/train.jsonl"
    val_file = "data/classifier/val.jsonl"
    train_and_evaluate(train_file, val_file)

if __name__ == "__main__":
    main()
    