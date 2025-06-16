import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

class InsuranceEDA:
    def __init__(self, df):
        self.df = df
        # Calculate metrics
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
        self.df['ClaimSeverity'] = self.df['TotalClaims'].where(self.df['TotalClaims'] > 0)
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']

    def descriptive_stats(self):
        """Calculate descriptive statistics for numerical features."""
        num_cols = ['TotalPremium', 'TotalClaims', 'LossRatio', 'ClaimSeverity', 'Margin']
        stats = self.df[num_cols].describe()
        stats.loc['variance'] = self.df[num_cols].var()
        stats.loc['coef_var'] = self.df[num_cols].std() / self.df[num_cols].mean()
        print("Descriptive Statistics:\n", stats)
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
        num_cols = ['TotalPremium', 'TotalClaims', 'LossRatio', 'ClaimSeverity', 'Margin']
        cat_cols = ['CoverType', 'make', 'Province', 'ZipCode', 'Gender']

        for col in num_cols:
            plt.figure(figsize=(10, 5))
            data = self.df[col].dropna()
            data = data[data > 0] if col in ['TotalPremium', 'TotalClaims', 'ClaimSeverity'] else data
            sns.histplot(np.log1p(data) if col in ['TotalPremium', 'TotalClaims', 'ClaimSeverity'] else data, 
                        kde=True, color='teal', bins=50)
            plt.title(f"Distribution of {col} {'(Log Scale)' if col in ['TotalPremium', 'TotalClaims', 'ClaimSeverity'] else ''}", fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.tight_layout()
            plt.show()

        for col in cat_cols:
            if col in self.df.columns:
                plt.figure(figsize=(10, 5))
                top_n = self.df[col].value_counts().head(10)
                sns.barplot(x=top_n.values, y=top_n.index, palette='viridis')
                plt.title(f"Top 10 {col} Distribution", fontsize=14)
                plt.xlabel("Count", fontsize=12)
                plt.ylabel(col, fontsize=12)
                plt.tight_layout()
                plt.show()

    def bivariate_analysis(self):
        """Analyze relationships between TotalPremium, TotalClaims, and ZipCode."""
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

        num_cols = ['TotalPremium', 'TotalClaims', 'LossRatio', 'ClaimSeverity', 'Margin']
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

        plt.figure(figsize=(12, 6))
        sns.barplot(data=self.df, x='Province', y='TotalPremium', hue='CoverType', palette='Set2')
        plt.xticks(rotation=45)
        plt.title("Average TotalPremium by Province and CoverType", fontsize=14)
        plt.xlabel("Province", fontsize=12)
        plt.ylabel("Average TotalPremium", fontsize=12)
        plt.tight_layout()
        plt.show()

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
        num_cols = ['TotalPremium', 'TotalClaims', 'ClaimSeverity', 'Margin']
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
        if 'ZipCode' in self.df.columns:
            top_zipcodes = self.df['ZipCode'].value_counts().head(5).index
            df_subset = self.df[self.df['ZipCode'].isin(top_zipcodes) & (self.df['TotalPremium'] > 0) & (self.df['TotalClaims'] >= 0)]
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=df_subset, x='TotalPremium', y='TotalClaims', hue='ZipCode', 
                            size='LossRatio', sizes=(50, 500), palette='viridis', alpha=0.8)
            plt.xscale('log')
            plt.yscale('log')
            plt.title("Premium vs Claims by ZipCode (Top 5)", fontsize=16, pad=20)
            plt.xlabel("Total Premium (Log Scale)", fontsize=14)
            plt.ylabel("Total Claims (Log Scale)", fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

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

        if 'Province' in self.df.columns:
            plt.figure(figsize=(12, 8))
            sns.violinplot(data=self.df, x='Province', y='LossRatio', palette='coolwarm', inner='quartile')
            plt.xticks(rotation=45)
            plt.title("Loss Ratio Distribution by Province", fontsize=16, pad=20)
            plt.xlabel("Province", fontsize=14)
            plt.ylabel("Loss Ratio", fontsize=14)
            plt.tight_layout()
            plt.show()

    def hypothesis_testing(self):
        """Conduct statistical tests for risk and margin hypotheses."""
        results = []

        # Helper function to check normality
        def check_normality(data, group_name, metric):
            stat, p = stats.shapiro(data.dropna()[:5000])  # Sample for large datasets
            return p > 0.05

        # H1: Risk differences across provinces
        if 'Province' in self.df.columns:
            # Claim Frequency: Chi-squared test
            province_freq = pd.crosstab(self.df['Province'], self.df['HasClaim'])
            chi2_freq, p_freq, _, _ = stats.chi2_contingency(province_freq)
            # Claim Severity: ANOVA or Kruskal-Wallis
            province_groups = [self.df[self.df['Province'] == p]['ClaimSeverity'].dropna() for p in self.df['Province'].unique()]
            if all(check_normality(g, f"Province {p}", "ClaimSeverity") for g, p in zip(province_groups, self.df['Province'].unique())):
                f_stat_sev, p_sev = stats.f_oneway(*[g for g in province_groups if len(g) > 0])
                test_sev = "ANOVA"
            else:
                h_stat_sev, p_sev = stats.kruskal(*[g for g in province_groups if len(g) > 0])
                test_sev = "Kruskal-Wallis"
            # Post-hoc for significant results
            tukey_sev = None
            if p_sev < 0.05:
                tukey_sev = pairwise_tukeyhsd(self.df['ClaimSeverity'].dropna(), self.df['Province'][self.df['ClaimSeverity'].notna()])
            results.append({
                'Hypothesis': 'No risk differences across provinces',
                'Metric': 'Claim Frequency',
                'Test': 'Chi-squared',
                'p-value': p_freq,
                'Reject H0': p_freq < 0.05,
                'Details': province_freq.to_dict()
            })
            results.append({
                'Hypothesis': 'No risk differences across provinces',
                'Metric': 'Claim Severity',
                'Test': test_sev,
                'p-value': p_sev,
                'Reject H0': p_sev < 0.05,
                'Details': tukey_sev.summary().as_text() if tukey_sev else 'No post-hoc needed'
            })

        # H2: Risk differences between zip codes (top 5 for simplicity)
        if 'ZipCode' in self.df.columns:
            top_zips = self.df['ZipCode'].value_counts().head(5).index
            df_zip = self.df[self.df['ZipCode'].isin(top_zips)]
            # Claim Frequency: Chi-squared test
            zip_freq = pd.crosstab(df_zip['ZipCode'], df_zip['HasClaim'])
            chi2_freq, p_freq, _, _ = stats.chi2_contingency(zip_freq)
            # Claim Severity: ANOVA or Kruskal-Wallis
            zip_groups = [df_zip[df_zip['ZipCode'] == z]['ClaimSeverity'].dropna() for z in top_zips]
            if all(check_normality(g, f"ZipCode {z}", "ClaimSeverity") for g, z in zip(zip_groups, top_zips)):
                f_stat_sev, p_sev = stats.f_oneway(*[g for g in zip_groups if len(g) > 0])
                test_sev = "ANOVA"
            else:
                h_stat_sev, p_sev = stats.kruskal(*[g for g in zip_groups if len(g) > 0])
                test_sev = "Kruskal-Wallis"
            tukey_sev = None
            if p_sev < 0.05:
                tukey_sev = pairwise_tukeyhsd(df_zip['ClaimSeverity'].dropna(), df_zip['ZipCode'][df_zip['ClaimSeverity'].notna()])
            results.append({
                'Hypothesis': 'No risk differences between zip codes',
                'Metric': 'Claim Frequency',
                'Test': 'Chi-squared',
                'p-value': p_freq,
                'Reject H0': p_freq < 0.05,
                'Details': zip_freq.to_dict()
            })
            results.append({
                'Hypothesis': 'No risk differences between zip codes',
                'Metric': 'Claim Severity',
                'Test': test_sev,
                'p-value': p_sev,
                'Reject H0': p_sev < 0.05,
                'Details': tukey_sev.summary().as_text() if tukey_sev else 'No post-hoc needed'
            })

        # H3: Margin differences between zip codes
        if 'ZipCode' in self.df.columns:
            if all(check_normality(df_zip[df_zip['ZipCode'] == z]['Margin'], f"ZipCode {z}", "Margin") for z in top_zips):
                f_stat_margin, p_margin = stats.f_oneway(*[df_zip[df_zip['ZipCode'] == z]['Margin'].dropna() for z in top_zips])
                test_margin = "ANOVA"
            else:
                h_stat_margin, p_margin = stats.kruskal(*[df_zip[df_zip['ZipCode'] == z]['Margin'].dropna() for z in top_zips])
                test_margin = "Kruskal-Wallis"
            tukey_margin = None
            if p_margin < 0.05:
                tukey_margin = pairwise_tukeyhsd(df_zip['Margin'].dropna(), df_zip['ZipCode'][df_zip['Margin'].notna()])
            results.append({
                'Hypothesis': 'No significant margin difference between zip codes',
                'Metric': 'Margin',
                'Test': test_margin,
                'p-value': p_margin,
                'Reject H0': p_margin < 0.05,
                'Details': tukey_margin.summary().as_text() if tukey_margin else 'No post-hoc needed'
            })

        # H4: Risk differences between Women and Men
        if 'Gender' in self.df.columns:
            df_gender = self.df[self.df['Gender'].isin(['Male', 'Female'])]
            # Claim Frequency: Chi-squared test
            gender_freq = pd.crosstab(df_gender['Gender'], df_gender['HasClaim'])
            chi2_freq, p_freq, _, _ = stats.chi2_contingency(gender_freq)
            # Claim Severity: t-test or Mann-Whitney U
            male_sev = df_gender[df_gender['Gender'] == 'Male']['ClaimSeverity'].dropna()
            female_sev = df_gender[df_gender['Gender'] == 'Female']['ClaimSeverity'].dropna()
            if check_normality(male_sev, "Male", "ClaimSeverity") and check_normality(female_sev, "Female", "ClaimSeverity"):
                t_stat_sev, p_sev = stats.ttest_ind(male_sev, female_sev, equal_var=False)
                test_sev = "t-test (Welch)"
            else:
                u_stat_sev, p_sev = stats.mannwhitneyu(male_sev, female_sev)
                test_sev = "Mann-Whitney U"
            results.append({
                'Hypothesis': 'No significant risk difference between Women and Men',
                'Metric': 'Claim Frequency',
                'Test': 'Chi-squared',
                'p-value': p_freq,
                'Reject H0': p_freq < 0.05,
                'Details': gender_freq.to_dict()
            })
            results.append({
                'Hypothesis': 'No significant risk difference between Women and Men',
                'Metric': 'Claim Severity',
                'Test': test_sev,
                'p-value': p_sev,
                'Reject H0': p_sev < 0.05,
                'Details': f'Male mean: {male_sev.mean():.2f}, Female mean: {female_sev.mean():.2f}'
            })

        # Report results
        print("\nHypothesis Testing Results:")
        for result in results:
            print(f"\nHypothesis: {result['Hypothesis']}")
            print(f"Metric: {result['Metric']}")
            print(f"Test: {result['Test']}")
            print(f"p-value: {result['p-value']:.4f}")
            print(f"Reject H0: {result['Reject H0']}")
            print(f"Details: {result['Details']}")

        return results