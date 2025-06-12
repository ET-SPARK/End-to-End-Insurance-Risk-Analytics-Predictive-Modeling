import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class InsuranceEDA:
    def __init__(self, df):
        self.df = df

    def descriptive_stats(self):
        """Calculate descriptive statistics for numerical features."""
        num_cols = ['TotalPremium', 'TotalClaims', 'LossRatio']
        stats = self.df[num_cols].describe()
        # Add variability measures: variance and coefficient of variation
        stats.loc['variance'] = self.df[num_cols].var()
        stats.loc['coef_var'] = self.df[num_cols].std() / self.df[num_cols].mean()
        print("Descriptive Statistics (including variance and coefficient of variation):\n", stats)
        return stats

    def data_types(self):
        """Review data types of all columns."""
        print("Data Types:\n", self.df.dtypes)

    def check_missing_values(self):
        """Check for missing values in all columns."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_pct})
        print("Missing Values:\n", missing_df)
        return missing_df

    def plot_univariate_distributions(self):
        """Plot histograms for numerical columns and bar charts for categorical columns."""
        num_cols = ['TotalPremium', 'TotalClaims', 'LossRatio']
        cat_cols = ['CoverType', 'make', 'Province', 'ZipCode']

        # Numerical distributions (log-scaled for skewed data)
        for col in num_cols:
            plt.figure(figsize=(10, 5))
            data = self.df[col].dropna()
            data = data[data > 0] if col in ['TotalPremium', 'TotalClaims'] else data
            sns.histplot(np.log1p(data) if col in ['TotalPremium', 'TotalClaims'] else data, 
                        kde=True, color='teal', bins=50)
            plt.title(f"Distribution of {col} {'(Log Scale)' if col in ['TotalPremium', 'TotalClaims'] else ''}", fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.tight_layout()
            plt.show()

        # Categorical distributions
        for col in cat_cols:
            if col in self.df.columns:
                plt.figure(figsize=(10, 5))
                top_n = self.df[col].value_counts().head(10)  # Top 10 for readability
                sns.barplot(x=top_n.values, y=top_n.index, palette='viridis')
                plt.title(f"Top 10 {col} Distribution", fontsize=14)
                plt.xlabel("Count", fontsize=12)
                plt.ylabel(col, fontsize=12)
                plt.tight_layout()
                plt.show()

    def bivariate_analysis(self):
        """Analyze relationships between TotalPremium, TotalClaims, and ZipCode."""
        # Scatter plot: TotalPremium vs TotalClaims by ZipCode (sample top 5 zip codes)
        if 'ZipCode' in self.df.columns:
            top_zipcodes = self.df['ZipCode'].value_counts().head(5).index
            df_subset = self.df[self.df['ZipCode'].isin(top_zipcodes)]
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=df_subset, x='TotalPremium', y='TotalClaims', hue='ZipCode', 
                           palette='deep', size='LossRatio', sizes=(20, 200), alpha=0.7)
            plt.title("TotalPremium vs TotalClaims by ZipCode (Top 5)", fontsize=14)
            plt.xlabel("TotalPremium", fontsize=12)
            plt.ylabel("TotalClaims", fontsize=12)
            plt.xscale('log')
            plt.yscale('log')
            plt.tight_layout()
            plt.show()

        # Correlation matrix
        num_cols = ['TotalPremium', 'TotalClaims', 'LossRatio']
        corr = self.df[num_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title("Correlation Matrix of Numerical Features", fontsize=14)
        plt.tight_layout()
        plt.show()

    def geo_trends(self):
        """Compare trends in CoverType, TotalPremium, and make over Province."""
        if 'Province' not in self.df.columns:
            print("Warning: 'Province' column not found. Skipping geo_trends.")
            return

        # Average TotalPremium by Province and CoverType
        plt.figure(figsize=(12, 6))
        sns.barplot(data=self.df, x='Province', y='TotalPremium', hue='CoverType', palette='Set2')
        plt.xticks(rotation=45)
        plt.title("Average TotalPremium by Province and CoverType", fontsize=14)
        plt.xlabel("Province", fontsize=12)
        plt.ylabel("Average TotalPremium", fontsize=12)
        plt.tight_layout()
        plt.show()

        # Count of policies by make and Province (top 5 makes)
        top_makes = self.df['make'].value_counts().head(5).index
        df_subset = self.df[self.df['make'].isin(top_makes)]
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df_subset, x='Province', hue='make', palette='husl')
        plt.xticks(rotation=45)
        plt.title("Policy Count by Province and Top 5 Vehicle Makes", fontsize=14)
        plt.xlabel("Province", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.tight_layout()
        plt.show()

    def boxplot_outliers(self):
        """Detect outliers in numerical columns using box plots."""
        num_cols = ['TotalPremium', 'TotalClaims']
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df[num_cols].apply(np.log1p), palette='Set2')
        plt.title("Log-Scaled Box Plot for Outlier Detection", fontsize=14)
        plt.ylabel("Log Value", fontsize=12)
        plt.tight_layout()
        plt.show()

    def time_trend(self, value_col: str):
        """Plot monthly trend of a numerical column."""
        if not pd.api.types.is_datetime64_any_dtype(self.df['TransactionMonth']):
            raise ValueError("TransactionMonth must be datetime. Run preprocessor first.")
        plt.figure(figsize=(12, 6))
        trend = self.df.groupby(self.df['TransactionMonth'].dt.to_period("M"))[value_col].sum().reset_index()
        trend['TransactionMonth'] = trend['TransactionMonth'].astype(str)
        sns.lineplot(data=trend, x='TransactionMonth', y=value_col, marker='o', color='steelblue')
        plt.xticks(rotation=45)
        plt.title(f"Monthly Trend of {value_col}", fontsize=14)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel(value_col, fontsize=12)
        plt.tight_layout()
        plt.show()

    def creative_plots(self):
        """Generate three creative and beautiful plots for key insights."""
        # Plot 1: Animated Bubble Plot for TotalPremium vs TotalClaims by ZipCode
        if 'ZipCode' in self.df.columns:
            top_zipcodes = self.df['ZipCode'].value_counts().head(5).index
            df_subset = self.df[self.df['ZipCode'].isin(top_zipcodes) & (self.df['TotalPremium'] > 0) & (self.df['TotalClaims'] >= 0)]
            plt.figure(figsize=(12, 8))
            scatter = sns.scatterplot(data=df_subset, x='TotalPremium', y='TotalClaims', hue='ZipCode', 
                                    size='LossRatio', sizes=(50, 500), palette='viridis', alpha=0.8)
            plt.xscale('log')
            plt.yscale('log')
            plt.title("Premium vs Claims by ZipCode (Top 5)", fontsize=16, pad=20)
            plt.xlabel("Total Premium (Log Scale)", fontsize=14)
            plt.ylabel("Total Claims (Log Scale)", fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

        # Plot 2: Stacked Bar Plot for CoverType Distribution Over Time
        if 'CoverType' in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df['TransactionMonth']):
            df_time = self.df.groupby([self.df['TransactionMonth'].dt.to_period("M"), 'CoverType']).size().unstack().fillna(0)
            df_time.index = df_time.index.astype(str)
            plt.figure(figsize=(12, 8))
            df_time.plot(kind='bar', stacked=True, colormap='Paired', figsize=(12, 8))
            plt.title("CoverType Distribution Over Time", fontsize=16, pad=20)
            plt.xlabel("Month", fontsize=14)
            plt.ylabel("Number of Policies", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="CoverType", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

        # Plot 3: Violin Violin Plot for LossRatio by Province
        if 'Province' in self.df.columns:
            plt.figure(figsize=(12, 8))
            sns.violinplot(data=self.df, x='Province', y='LossRatio', palette='coolwarm', inner='quartile')
            plt.xticks(rotation=45)
            plt.title("Loss Ratio Distribution by Province", fontsize=16, pad=20)
            plt.xlabel("Province", fontsize=14)
            plt.ylabel("Loss Ratio", fontsize=14)
            plt.tight_layout()
            plt.show()