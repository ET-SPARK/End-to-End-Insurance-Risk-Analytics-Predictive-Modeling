import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import shap
import matplotlib.pyplot as plt
import os

class ModelEvaluator:
    def __init__(self, data_dir="data/prepared", model_dir="models", results_dir="results"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self):
        """Load all prepared datasets."""
        self.severity_data = {
            "X_train": pd.read_csv(f"{self.data_dir}/X_sev_train.csv"),
            "X_test": pd.read_csv(f"{self.data_dir}/X_sev_test.csv"),
            "y_train": pd.read_csv(f"{self.data_dir}/y_sev_train.csv").squeeze(),
            "y_test": pd.read_csv(f"{self.data_dir}/y_sev_test.csv").squeeze()
        }
        self.prob_data = {
            "X_train": pd.read_csv(f"{self.data_dir}/X_prob_train.csv"),
            "X_test": pd.read_csv(f"{self.data_dir}/X_prob_test.csv"),
            "y_train": pd.read_csv(f"{self.data_dir}/y_prob_train.csv").squeeze(),
            "y_test": pd.read_csv(f"{self.data_dir}/y_prob_test.csv").squeeze()
        }
        self.premium_data = {
            "X_train": pd.read_csv(f"{self.data_dir}/X_prem_train.csv"),
            "X_test": pd.read_csv(f"{self.data_dir}/X_prem_test.csv"),
            "y_train": pd.read_csv(f"{self.data_dir}/y_prem_train.csv").squeeze(),
            "y_test": pd.read_csv(f"{self.data_dir}/y_prem_test.csv").squeeze()
        }

    def evaluate_regression(self, model, X_test, y_test, model_name):
        """Evaluate regression model."""
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return {"Model": model_name, "RMSE": rmse, "R2": r2}

    def evaluate_classification(self, model, X_test, y_test, model_name):
        """Evaluate classification model."""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }

    def shap_analysis(self, model, X, model_name, task, top_n=10):
        """Perform SHAP analysis."""
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=top_n, show=False)
        plt.title(f"SHAP Feature Importance for {model_name} ({task})")
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/shap_{model_name}_{task}.png")
        plt.close()

    def evaluate_all(self):
        """Evaluate all models and save results."""
        results = []

        # Claim Severity Models
        for name in ["LinearRegression", "RandomForest", "XGBoost"]:
            model = joblib.load(f"{self.model_dir}/{name}_severity.pkl")
            result = self.evaluate_regression(
                model, self.severity_data["X_test"], self.severity_data["y_test"], f"{name}_Severity"
            )
            results.append(result)
            if name == "XGBoost":  # SHAP for best model
                self.shap_analysis(model, self.severity_data["X_test"], name, "Severity")

        # Claim Probability Models
        for name in ["LogisticRegression", "RandomForest", "XGBoost"]:
            model = joblib.load(f"{self.model_dir}/{name}_probability.pkl")
            result = self.evaluate_classification(
                model, self.prob_data["X_test"], self.prob_data["y_test"], f"{name}_Probability"
            )
            results.append(result)
            if name == "XGBoost":  # SHAP for best model
                self.shap_analysis(model, self.prob_data["X_test"], name, "Probability")

        # Premium Models
        for name in ["LinearRegression", "RandomForest", "XGBoost", "RiskBasedPremium"]:
            model = joblib.load(f"{self.model_dir}/{name}_premium.pkl")
            result = self.evaluate_regression(
                model, self.premium_data["X_test"], self.premium_data["y_test"], f"{name}_Premium"
            )
            results.append(result)
            if name == "XGBoost":  # SHAP for best model
                self.shap_analysis(model, self.premium_data["X_test"], name, "Premium")

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{self.results_dir}/evaluation_metrics.csv", index=False)
        print("Evaluation results saved to evaluation_metrics.csv")

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.load_data()
    evaluator.evaluate_all()