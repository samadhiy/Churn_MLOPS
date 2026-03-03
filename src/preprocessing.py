import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess():
    df = pd.read_csv("data/processed/raw_copy.csv")

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID
    df.drop("customerID", axis=1, inplace=True)

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    os.makedirs("data/processed", exist_ok=True)

    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

if __name__ == "__main__":
    preprocess()