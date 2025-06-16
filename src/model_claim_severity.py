import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

class ClaimSeverityModel:
    def __init__(self, data_dir="data/prepared", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
        }

    def load_data(self):
        """Load prepared datasets."""
        self.X_train = pd.read_csv(f"{self.data_dir}/X_sev_train.csv")
        self.X_test = pd.read_csv(f"{self.data_dir}/X_sev_test.csv")
        self.y_train = pd.read_csv(f"{self.data_dir}/y_sev_train.csv").squeeze()
        self.y_test = pd.read_csv(f"{self.data_dir}/y_sev_test.csv").squeeze()

    def train_models(self):
        """Train all models."""
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            joblib.dump(model, f"{self.model_dir}/{name}_severity.pkl")
            print(f"Trained and saved {name} model.")

    def run_all(self):
        self.load_data()
        self.train_models()

if __name__ == "__main__":
    model = ClaimSeverityModel()
    model.run_all()