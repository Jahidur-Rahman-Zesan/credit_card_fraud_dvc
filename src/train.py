import joblib
import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


def train():
    # Read parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    # Load processed training data
    train_df = pd.read_csv("data/processed/train.csv")
    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]

    # Train model
    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )
    clf.fit(X_train, y_train)

    # Save model binary
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "model.pkl")
    print("Training completed successfully.")


if __name__ == "__main__":
    train()