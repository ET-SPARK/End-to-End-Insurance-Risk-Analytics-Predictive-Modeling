import pandas as pd
import numpy as np

class InsurancePreprocessor:
    def __init__(self, filepath, sep="|"):
        self.filepath = filepath
        self.sep = sep
        self.df = pd.read_csv(filepath, sep=self.sep)

    def clean_data(self):
        # Strip whitespace from column names
        self.df.columns = [col.strip() for col in self.df.columns]
        print("Columns after stripping:", self.df.columns.tolist())

        # Remove duplicates
        self.df.drop_duplicates(inplace=True)

        # Check for required columns
        required_cols = ["TotalPremium", "TotalClaims", "TransactionMonth"]
        for col in required_cols:
            if col not in self.df.columns:
                raise KeyError(f"Missing required column: {col}")

        # Drop rows with missing required columns
        self.df.dropna(subset=required_cols, inplace=True)

        # Convert to numeric and datetime
        self.df["TotalClaims"] = pd.to_numeric(self.df["TotalClaims"], errors="coerce")
        self.df["TotalPremium"] = pd.to_numeric(self.df["TotalPremium"], errors="coerce")
        # Convert TransactionMonth to datetime
        self.df["TransactionMonth"] = pd.to_datetime(self.df["TransactionMonth"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        # Debug: Check dtype and sample values
        print("TransactionMonth dtype after conversion:", self.df["TransactionMonth"].dtype)
        print("Sample TransactionMonth values:", self.df["TransactionMonth"].head().tolist())
        print("Number of NaT values in TransactionMonth:", self.df["TransactionMonth"].isnull().sum())

        # Drop rows with NaT in TransactionMonth
        nat_count = self.df["TransactionMonth"].isnull().sum()
        if nat_count > 0:
            print(f"Warning: {nat_count} rows with NaT in TransactionMonth. Dropping these rows.")
            self.df.dropna(subset=["TransactionMonth"], inplace=True)

        # Calculate LossRatio, avoiding division by zero
        self.df["LossRatio"] = self.df.apply(
            lambda x: x["TotalClaims"] / x["TotalPremium"] if x["TotalPremium"] != 0 else 0, axis=1
        )
        print("Preprocessing complete. 'LossRatio' column added.")

    def save_cleaned_data(self, output_path):
        self.df.to_csv(output_path, index=False)

    def run_all(self, output_path):
        self.clean_data()
        self.save_cleaned_data(output_path)