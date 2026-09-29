import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feast import FeatureStore


def evaluate():
    model_path = Path("models/model.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found at models/model.pkl")
    model = joblib.load(model_path)

    store = FeatureStore(repo_path="feature_repo/feature_repo")
    
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # Offset test transaction_ids so they match full_data.parquet
    start_id = len(train_df) + 1
    test_df["transaction_id"] = range(start_id, start_id + len(test_df))
    test_df["event_timestamp"] = pd.to_datetime("2026-01-01 00:00:00", utc=True)

    entity_df = test_df[["transaction_id", "event_timestamp", "Class"]].copy()

    feature_list = [f"credit_card_features:V{i}" for i in range(1, 29)] + ["credit_card_features:Amount"]

    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_list,
    ).to_df().dropna()

    X_test = feature_df.drop(columns=["transaction_id", "event_timestamp", "Class"])
    y_test = feature_df["Class"]

    y_probs = model.predict_proba(X_test)[:, 1]

    # Apply a custom decision threshold (e.g., 0.70 instead of default 0.50)
    threshold = 0.70
    y_pred = (y_probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    metrics_path = Path("metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation completed successfully. Metrics saved to metrics.json")


if __name__ == "__main__":
    evaluate()