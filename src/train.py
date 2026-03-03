import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

mlflow.set_experiment("Customer_Churn_Experiment")

def train():

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train.values.ravel())

            mlflow.log_param("model_name", name)
            mlflow.sklearn.log_model(model, name)

            joblib.dump(model, f"models/{name}.pkl")

if __name__ == "__main__":
    train()