# 💳 Credit Card Fraud Detection Pipeline

An end-to-end MLOps pipeline for Credit Card Fraud Detection built with **DVC** for data versioning & experiment tracking, and **Feast** for feature store management and point-in-time correct feature serving.

---

## 🏗️ Architecture & Stack

* **Feature Store:** [Feast](https://feast.dev/) (Offline store for training feature retrieval)
* **Pipeline & Versioning:** [DVC](https://dvc.org/) (Data pipelines, experiment tracking, workspace state control)
* **Model Framework:** [Scikit-Learn](https://scikit-learn.org/) (Random Forest & Logistic Regression)
* **Config Management:** `params.yaml` (Hyperparameters & pipeline configuration)
* **Environment:** Python 3.13 / Windows PowerShell compatible

---

## 📁 Repository Structure

```text
credit_card_fraud_dvc/
├── .dvc/                   # DVC configuration and storage metadata
├── data/                   # Raw and processed datasets (Git-ignored)
│   ├── raw/
│   └── processed/
├── feature_repo/           # Feast Feature Store repository
│   ├── feature_store.yaml  # Feast registry and provider config
│   ├── credit_card_features.py  # Entity & FeatureView definitions
│   └── data/               # Local registry (.pb) and SQLite online store
├── models/                 # Trained model artifacts (.pkl)
├── src/                    # Source code
│   ├── prepare.py          # Data preprocessing stage
│   ├── train.py            # Model training & Feast feature loading
│   └── evaluate.py         # Evaluation & metrics generation
├── dvc.yaml                # DVC pipeline stage definitions
├── dvc.lock                # Pipeline execution lock file
├── params.yaml             # Config parameters (model_type, hyperparams)
├── metrics.json            # Target performance metrics
└── README.md


git clone [https://github.com/your-username/credit_card_fraud_dvc.git](https://github.com/your-username/credit_card_fraud_dvc.git)
cd credit_card_fraud_dvc
pip install -r requirements.txt


cd feature_repo
python -c "from feast.cli.cli import cli; cli.main(args=['apply'])"
cd ..
