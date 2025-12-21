#!/usr/bin/env python3
"""
Simple training script for MLflow Project.
- Expects a CSV dataset with features and a target column named 'target' OR
- If dataset not found/invalid, will train on iris dataset (from sklearn).
Saves model to output dir (mlflow.sklearn.save_model) and logs params/metrics.
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib

def load_data(path):
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Expect a column named 'target' or 'label'. Try common names.
            if 'target' in df.columns:
                X = df.drop(columns=['target'])
                y = df['target']
                return X, y
            if 'label' in df.columns:
                X = df.drop(columns=['label'])
                y = df['label']
                return X, y
            # If last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            return X, y
        except Exception as e:
            print(f"Failed to read dataset at {path}: {e}")
    # Fallback to synthetic iris
    print("Using fallback iris dataset.")
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    return X, y

def main(args):
    mlflow.set_experiment("Workflow-CI-Experiment")
    with mlflow.start_run():
        X, y = load_data(args.data_path)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.random_state, stratify=y if len(np.unique(y))>1 else None
        )

        model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log params and metrics
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_metric("accuracy", float(acc))

        # Ensure output dir exists
        os.makedirs(args.output, exist_ok=True)

        # Save model locally (as mlflow model)
        mlflow.sklearn.save_model(sk_model=model, path=args.output)

        # Also log model as artifact in mlflow
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        # Save a pickled copy
        joblib.dump(model, os.path.join(args.output, "model.pkl"))

        print(f"Training finished. Accuracy: {acc:.4f}")
        print(f"Model saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="namadataset_preprocessing/dataset.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output", type=str, default="artifacts/model")
    args = parser.parse_args()
    main(args)
