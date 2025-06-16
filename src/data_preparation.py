import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

class DataPreparator:
    def __init__(self, input_path):
        self.df = pd.read_csv(input_path)
        self.output_dir = "data/prepared"
        os.makedirs(self.output_dir, exist_ok=True)

    def handle_missing_data(self):
        """Impute or drop missing values."""
        # Numerical columns: impute with median
        num_cols = ['TotalPremium', 'TotalClaims', 'LossRatio']
        for col in num_cols:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Categorical columns: impute with mode
        cat_cols = ['Province', 'ZipCode', 'CoverType', 'make', 'Gender']
        for col in cat_cols:
            if col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

    def feature_engineering(self):
        """Create new features."""
        # Vehicle age (assuming 'VehicleAge' or proxy exists; adjust as needed)
        if 'VehicleAge' not in self.df.columns and 'TransactionMonth' in self.df.columns:
            self.df['VehicleAge'] = (pd.to_datetime(self.df['TransactionMonth']).dt.year - 2000).clip(lower=0)
        
        # Claim frequency indicator
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
        
        # Premium-to-claim ratio (already exists as LossRatio)
        self.df['PremiumPerClaim'] = self.df['TotalPremium'] / (self.df['TotalClaims'] + 1e-6)  # Avoid division by zero
        
        # Transaction month features
        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionYear'] = pd.to_datetime(self.df['TransactionMonth']).dt.year
            self.df['TransactionMonthOfYear'] = pd.to_datetime(self.df['TransactionMonth']).dt.month

    def encode_categorical(self):
        """Encode categorical variables using label encoding."""
        cat_cols = ['Province', 'ZipCode', 'CoverType', 'make', 'Gender']
        self.label_encoders = {}
        for col in cat_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le

    def prepare_datasets(self):
        """Split data into train/test for each modeling task."""
        # Claim Severity (Regression: TotalClaims where HasClaim == 1)
        severity_df = self.df[self.df['HasClaim'] == 1].copy()
        X_severity = severity_df.drop(['TotalClaims', 'HasClaim', 'TransactionMonth'], axis=1, errors='ignore')
        X_severity = X_severity.select_dtypes(include=[np.number])  # Only numeric features
        y_severity = severity_df['TotalClaims']
        X_sev_train, X_sev_test, y_sev_train, y_sev_test = train_test_split(
            X_severity, y_severity, test_size=0.2, random_state=42
        )
        
        # Claim Probability (Classification: HasClaim)
        X_prob = self.df.drop(['HasClaim', 'TotalClaims', 'TransactionMonth'], axis=1, errors='ignore')
        X_prob = X_prob.select_dtypes(include=[np.number])
        y_prob = self.df['HasClaim']
        X_prob_train, X_prob_test, y_prob_train, y_prob_test = train_test_split(
            X_prob, y_prob, test_size=0.2, random_state=42
        )
        
        # Premium Optimization (Regression: TotalPremium)
        X_prem = self.df.drop(['TotalPremium', 'HasClaim', 'TotalClaims', 'TransactionMonth'], axis=1, errors='ignore')
        X_prem = X_prem.select_dtypes(include=[np.number])
        y_prem = self.df['TotalPremium']
        X_prem_train, X_prem_test, y_prem_train, y_prem_test = train_test_split(
            X_prem, y_prem, test_size=0.2, random_state=42
        )

        # Save datasets
        pd.DataFrame(X_sev_train).to_csv(f"{self.output_dir}/X_sev_train.csv", index=False)
        pd.DataFrame(X_sev_test).to_csv(f"{self.output_dir}/X_sev_test.csv", index=False)
        y_sev_train.to_csv(f"{self.output_dir}/y_sev_train.csv", index=False)
        y_sev_test.to_csv(f"{self.output_dir}/y_sev_test.csv", index=False)
        
        pd.DataFrame(X_prob_train).to_csv(f"{self.output_dir}/X_prob_train.csv", index=False)
        pd.DataFrame(X_prob_test).to_csv(f"{self.output_dir}/X_prob_test.csv", index=False)
        y_prob_train.to_csv(f"{self.output_dir}/y_prob_train.csv", index=False)
        y_prob_test.to_csv(f"{self.output_dir}/y_prob_test.csv", index=False)
        
        pd.DataFrame(X_prem_train).to_csv(f"{self.output_dir}/X_prem_train.csv", index=False)
        pd.DataFrame(X_prem_test).to_csv(f"{self.output_dir}/X_prem_test.csv", index=False)
        y_prem_train.to_csv(f"{self.output_dir}/y_prem_train.csv", index=False)
        y_prem_test.to_csv(f"{self.output_dir}/y_prem_test.csv", index=False)

    def run_all(self):
        self.handle_missing_data()
        self.feature_engineering()
        self.encode_categorical()
        self.prepare_datasets()

if __name__ == "__main__":
    preparator = DataPreparator("data/insurance_clean.csv")
    preparator.run_all()