import pandas as pd
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    # Load the CSV with the new name
    df = load_data("data/raw/Churn Prediction Dataset.csv")
    
    # Ensure the processed folder exists
    os.makedirs("data/processed", exist_ok=True)
    
    # Save a copy in the processed folder
    df.to_csv("data/processed/raw_copy.csv", index=False)