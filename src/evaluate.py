import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate():
    # Load test data and trained model
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    model = joblib.load("models/model.pkl")

    # Evaluate predictions
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Save evaluation metrics
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation completed successfully.")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    evaluate()