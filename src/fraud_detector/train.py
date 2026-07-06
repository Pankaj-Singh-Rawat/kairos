'''
using random forest classifier: votes among different trees and makes decisions based on it.
'''

import pandas as pd
import numpy as np
from mlflow.entities.model_registry import registered_model_tag
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score
import mlflow
import mlflow.sklearn

def prepare_data(file_path: str, max_step: int = 100):
    df = pd.read_csv(file_path)
    df = df[df['step']<= max_step].copy() # filtering for baseline periods

    # categorical encoding 'type'
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # drop columns that won't generalize well or are labels
    features_to_drop = ['step', 'nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
    X = df.drop(columns=features_to_drop, errors="ignore")
    y = df['isFraud']

    # ensuring all boolean columns from get_dummies are numeric
    X = X.astype(float)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_baseline():
    mlflow.set_experiment("Kairos_Fraud_Detection")

    with mlflow.start_run(run_name="Baseline_Model") as run:
        print("Preparing Baseline data ... ")
        X_train, X_val, y_train, y_val = prepare_data("data/paysim.csv")

        # Hyperparameters
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        mlflow.log_params(params)

        print("Training Random Forest model ...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # evaluate
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        has_fraud = 1 in y_val.values

        metrics = {
            "roc_auc": roc_auc_score(y_val, probs) if has_fraud else 0.5,
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds, zero_division=0)
        }
        mlflow.log_metrics(metrics)
        print(f"Logged Metrics: {metrics}")

        # log and register the model
        # using input_examples helps mlflow understand the feature schema
        input_example = X_train.head(1)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="fraud_model",
            registered_model_name = "Kairos_Fraud_Baseline",
            input_example = input_example
        )

        print("Model successfully registered in MLflow.")

if __name__ == "__main__":
    train_baseline()