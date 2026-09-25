import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess():
    # Read parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["preprocess"]

    # Load dataset
    data = pd.read_csv("data/raw/dataset.csv")

    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Recombine features and targets into DataFrames
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df["Class"] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df["Class"] = y_test.values

    # Save outputs
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess()