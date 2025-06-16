import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os

class PremiumModel:
    def __init__(self, data_dir="data/prepared", model_dir="models", severity_model_dir="models", prob_model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.severity_model_dir = severity_model_dir
        self.prob_model_dir = prob_model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
        }
        self.risk_based_model = None

    def load_data(self):
        """Load prepared datasets."""
        self.X_train = pd.read_csv(f"{self.data_dir}/X_prem_train.csv")
        self.X_test = pd.read_csv(f"{self.data_dir}/X_prem_test.csv")
        self.y_train = pd.read_csv(f"{self.data_dir}/y_prem_train.csv").squeeze()
        self.y_test = pd.read_csv(f"{self.data_dir}/y_prem_test.csv").squeeze()

    def load_risk_models(self):
        """Load severity and probability models for risk-based pricing."""
        self.severity_model = joblib.load(f"{self.severity_model_dir}/XGBoost_severity.pkl")
        self.prob_model = joblib.load(f"{self.prob_model_dir}/XGBoost_probability.pkl")

    def train_risk_based_premium(self):
        """Calculate risk-based premium: (Prob of Claim * Predicted Severity) + Expense + Profit."""
        # Predict claim probability and severity
        claim_probs = self.prob_model.predict_proba(self.X_train)[:, 1]
        claim_severity = self.severity_model.predict(self.X_train)
        
        # Assume expense loading (10%) and profit margin (5%)
        expense_loading = 0.10
        profit_margin = 0.05
        risk_premium = claim_probs * claim_severity
        total_premium = risk_premium * (1 + expense_loading + profit_margin)
        
        # Train a model to predict this risk-based premium
        self.risk_based_model = XGBRegressor(n_estimators=100, random_state=42)
        self.risk_based_model.fit(self.X_train, total_premium)
        joblib.dump(self.risk_based_model, f"{self.model_dir}/RiskBasedPremium.pkl")
        print("Trained and saved RiskBasedPremium model.")

    def train_models(self):
        """Train all models."""
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            joblib.dump(model, f"{self.model_dir}/{name}_premium.pkl")
            print(f"Trained and saved {name} model.")

    def run_all(self):
        self.load_data()
        self.load_risk_models()
        self.train_models()
        self.train_risk_based_premium()

if __name__ == "__main__":
    model = PremiumModel()
    model.run_all()