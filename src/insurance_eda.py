import pandas as pd  # Added import
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class InsuranceEDA:
    def __init__(self, df):
        self.df = df

    def descriptive_stats(self):
        print(self.df.describe())

    def data_types(self):
        print(self.df.dtypes)

    def check_missing_values(self):
        print(self.df.isnull().sum())

    def plot_distributions(self):
        num_cols = ['TotalPremium', 'TotalClaims', 'LossRatio']
        for col in num_cols:
            plt.figure(figsize=(10, 5))
            data = self.df[col].dropna()
            if col in ['TotalPremium', 'TotalClaims']:
                data = data[data > 0]  # Filter positive values for log transform
                sns.histplot(np.log(data), kde=True, color='steelblue')
                plt.title(f"Log Distribution of {col}")
            else:
                sns.histplot(data, kde=True, color='steelblue')
                plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    def boxplot_outliers(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df[['TotalPremium', 'TotalClaims']].apply(lambda x: np.log(x + 1)), palette='Set2')
        plt.title("Log-Scaled Outlier Detection for TotalPremium and TotalClaims")
        plt.tight_layout()
        plt.show()

    def geo_analysis(self):
        if 'Province' not in self.df.columns:
            print("Warning: 'Province' column not found. Skipping geo_analysis.")
            return
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Province', y='LossRatio', data=self.df, palette='viridis')
        plt.xticks(rotation=45)
        plt.title("Loss Ratio by Province")
        plt.tight_layout()
        plt.show()

    def vehicle_make_claims(self):
        if 'make' not in self.df.columns:
            print("Warning: 'make' column not found. Skipping vehicle_make_claims.")
            return
        top_makes = self.df.groupby('make')['TotalClaims'].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_makes.values, y=top_makes.index, color='salmon')  # Fixed earlier
        plt.title("Top 10 Vehicle Makes by Average TotalClaims")
        plt.xlabel("Average TotalClaims")
        plt.ylabel("Make")
        plt.tight_layout()
        plt.show()

    def time_trend(self, value_col: str):
        if not pd.api.types.is_datetime64_any_dtype(self.df['TransactionMonth']):
            raise ValueError("TransactionMonth must be datetime. Run preprocessor first.")
        plt.figure(figsize=(12, 6))
        trend = self.df.groupby(self.df['TransactionMonth'].dt.to_period("M"))[value_col].sum().reset_index()
        trend['TransactionMonth'] = trend['TransactionMonth'].astype(str)
        sns.lineplot(data=trend, x='TransactionMonth', y=value_col, marker='o', color='steelblue')
        plt.xticks(rotation=45)
        plt.title(f"Monthly Trend of {value_col}")
        plt.xlabel("Month")
        plt.ylabel(value_col)
        plt.tight_layout()
        plt.show()
        
 