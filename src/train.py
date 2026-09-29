import joblib
import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from feast import FeatureStore


def train():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    store = FeatureStore(repo_path="feature_repo/feature_repo")

    train_df = pd.read_csv("data/processed/train.csv")
    
    # Matching the exact ID offset generated for full_data.parquet
    train_df["transaction_id"] = range(1, len(train_df) + 1)
    train_df["event_timestamp"] = pd.to_datetime("2026-01-01 00:00:00", utc=True)

    entity_df = train_df[["transaction_id", "event_timestamp", "Class"]].copy()

    # Request all V1-V28 + Amount features
    feature_list = [f"credit_card_features:V{i}" for i in range(1, 29)] + ["credit_card_features:Amount"]

    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_list,
    ).to_df().dropna()

    X_train = feature_df.drop(columns=["transaction_id", "event_timestamp", "Class"])
    y_train = feature_df["Class"]

    model_type = params.get("model_type", "random_forest")

    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            class_weight="balanced",
        )
    elif model_type == "logistic_regression":
        clf = LogisticRegression(random_state=params["random_state"], class_weight="balanced")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    clf.fit(X_train, y_train)

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "model.pkl")
    print("Training completed successfully.")


if __name__ == "__main__":
    train()