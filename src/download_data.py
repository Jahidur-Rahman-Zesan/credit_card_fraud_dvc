from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_fraud_dataset(
    n_samples=5000, output_path="data/raw/dataset.csv"
):
    np.random.seed(42)

    # Convert to Path object to handle Windows/Mac/Linux automatically
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate synthetic binary classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        weights=[0.95, 0.05],  # 5% fraud cases
        random_state=42,
    )

    feature_names = [f"V{i+1}" for i in range(8)] + ["Amount", "Time"]
    df = pd.DataFrame(X, columns=feature_names)
    df["Class"] = y

    # Scale features
    df["Amount"] = np.round(np.abs(df["Amount"]) * 50, 2)
    df["Time"] = np.round(np.abs(df["Time"]) * 100, 0)

    df.to_csv(out_file, index=False)
    print(
        f"Dataset successfully created at {out_file.resolve()} with shape {df.shape}"
    )


if __name__ == "__main__":
    generate_fraud_dataset()
    