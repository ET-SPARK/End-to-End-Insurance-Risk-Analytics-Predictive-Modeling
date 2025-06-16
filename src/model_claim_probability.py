import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

class ClaimProbabilityModel:
    def __init__(self, data_dir="data/prepared", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42)
        }

    def load_data(self):
        """Load prepared datasets."""
        self.X_train = pd.read_csv(f"{self.data_dir}/X_prob_train.csv")
        self.X_test = pd.read_csv(f"{self.data_dir}/X_prob_test.csv")
        self.y_train = pd.read_csv(f"{self.data_dir}/y_prob_train.csv").squeeze()
        self.y_test = pd.read_csv(f"{self.data_dir}/y_prob_test.csv").squeeze()

    def train_models(self):
        """Train all models."""
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            joblib.dump(model, f"{self.model_dir}/{name}_probability.pkl")
            print(f"Trained and saved {name} model.")

    def run_all(self):
        self.load_data()
        self.train_models()

if __name__ == "__main__":
    model = ClaimProbabilityModel()
    model.run_all()