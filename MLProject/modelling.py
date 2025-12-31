import os
import sys
import argparse
import logging

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise KeyError("The dataframe does not contain a 'target' column.")
    return df

def main(data_path, n_estimators, random_state, output):
    try:
        df = load_data(data_path)

        X = df.drop("target", axis=1)
        y = df["target"]

        # Convert continuous target to discrete binary labels
        y = (y > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        # Enable autologging for scikit-learn models
        mlflow.sklearn.autolog()

        with mlflow.start_run(run_name="baseline_random_forest"):
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Log metrics (autolog may already capture some, but explicit logging is fine)
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("precision", float(prec))
            mlflow.log_metric("recall", float(rec))
            mlflow.log_metric("f1_score", float(f1))

            print(f"Accuracy : {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall   : {rec:.4f}")
            print(f"F1-score : {f1:.4f}")

            # Save model to output directory if specified
            if output:
                os.makedirs(output, exist_ok=True)
                mlflow.sklearn.save_model(model, output)
                logger.info(f"Model saved to {output}")

    except Exception as e:
        logger.exception("Failed to run modelling script")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline random forest model")
    parser.add_argument("--data_path", type=str, default="namadataset_preprocessing/dataset.csv",
                        help="path to dataset CSV")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="number of trees in random forest")
    parser.add_argument("--random_state", type=int, default=42,
                        help="random state for reproducibility")
    parser.add_argument("--output", type=str, default="artifacts/model",
                        help="output directory for model artifacts")
    args = parser.parse_args()
    main(args.data_path, args.n_estimators, args.random_state, args.output)
