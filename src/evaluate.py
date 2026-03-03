import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def evaluate():

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    models = ["LogisticRegression", "RandomForest", "XGBoost"]

    for name in models:
        model = joblib.load(f"models/{name}.pkl")

        with mlflow.start_run(run_name=f"{name}_evaluation"):

            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:,1]

            mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
            mlflow.log_metric("precision", precision_score(y_test, preds))
            mlflow.log_metric("recall", recall_score(y_test, preds))
            mlflow.log_metric("f1_score", f1_score(y_test, preds))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))

if __name__ == "__main__":
    evaluate()