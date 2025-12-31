
import mlflow
import mlflow.sklearn
import pandas as pd
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# Load dataset
dataset_path = "content/dataset_preprocessed (1).csv"

# Check if dataset file exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset file '{dataset_path}' not found.")
    sys.exit(1)

try:
    df = pd.read_csv(dataset_path)
except Exception as e:
    print(f"Error reading dataset: {e}")
    sys.exit(1)

# Verify 'target' column exists
if "target" not in df.columns:
    print(f"Error: 'target' column not found in dataset. Available columns: {list(df.columns)}")
    sys.exit(1)

X = df.drop("target", axis=1)
y = df["target"]

# Convert continuous target to discrete binary labels before splitting
y = (y > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Enable autologging for scikit-learn models
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="baseline_logistic_regression"): # autolog will log within this run
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log F1-score manually (other metrics are logged by autolog)
    mlflow.log_metric("f1_score", f1)

    print("=" * 40)
    print("Model Performance Metrics:")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 40)
