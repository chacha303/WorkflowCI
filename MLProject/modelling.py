
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("/content/dataset_preprocessed (1).csv")

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

    print("Accuracy:", acc)
