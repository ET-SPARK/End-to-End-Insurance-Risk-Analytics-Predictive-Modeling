# End-to-End-Insurance-Risk-Analytics-Predictive-Modeling

## Objective

AlphaCare Insurance Solutions (ACIS) is committed to developing cutting-edge risk and predictive analytics in the area of car insurance planning and marketing in South Africa. The first project is to analyze historical insurance claim data. The objective of this analysis is to help optimize the marketing strategy and discover “low-risk” targets for which the premium could be reduced, providing an opportunity to attract new clients.

---

## Task 1: Git & GitHub Setup and Project Planning - EDA & Stats

### Methodology

- **Data Source**: Loaded historical insurance data from `data/MachineLearningRating_v3.txt`, a pipe-separated (`|`) file containing columns such as `TotalPremium`, `TotalClaims`, `TransactionMonth`, `make`, `Province`, `ZipCode`, and `CoverType`.
- **Preprocessing Steps** (implemented in `src/preprocessing.py`):
  - Stripped whitespace from column names.
  - Removed duplicate records.
  - Ensured required columns (`TotalPremium`, `TotalClaims`, `TransactionMonth`) are present.
  - Converted `TotalClaims` and `TotalPremium` to numeric (`float64`), handling invalid values as `NaN`.
  - Converted `TransactionMonth` to datetime (`YYYY-MM-DD HH:MM:SS` format), dropping rows with invalid dates (`NaT`).
  - Calculated `LossRatio` as `TotalClaims / TotalPremium`, setting to 0 where `TotalPremium` is 0 to avoid division by zero.
  - Saved cleaned data to `data/insurance_clean.csv`.
- **Exploratory Data Analysis** (implemented in `src/insurance_eda.py`):
  - **Data Summarization**:
    - Calculated descriptive statistics (mean, std, quartiles, variance, coefficient of variation) for `TotalPremium`, `TotalClaims`, and `LossRatio`.
    - Reviewed data types to confirm correct formats (e.g., `TransactionMonth` as `datetime64[ns]`).
  - **Data Quality**:
    - Assessed missing values and their percentages for all columns.
  - **Univariate Analysis**:
    - Plotted log-scaled histograms for `TotalPremium`, `TotalClaims`, and `LossRatio`.
    - Plotted bar charts for categorical columns (`CoverType`, `make`, `Province`, `ZipCode`), showing top 10 categories.
  - **Bivariate/Multivariate Analysis**:
    - Created scatter plots of `TotalPremium` vs. `TotalClaims` by `ZipCode` (top 5), with `LossRatio` as bubble size.
    - Generated correlation matrix heatmap for numerical features.
  - **Geographic Trends**:
    - Analyzed `TotalPremium` by `Province` and `CoverType` using bar plots.
    - Visualized policy counts by `Province` and top 5 vehicle `make` values.
  - **Outlier Detection**:
    - Used log-scaled box plots to identify outliers in `TotalPremium` and `TotalClaims`.
  - **Creative Visualizations**:
    - Bubble plot: `TotalPremium` vs. `TotalClaims` by `ZipCode`, highlighting high-risk areas.
    - Stacked bar plot: `CoverType` distribution over time, showing market trends.
    - Violin plot: `LossRatio` distribution by `Province`, revealing regional risk variability.
- **Execution**: Ran preprocessing and EDA via `main.py`, which loads data, preprocesses it, and calls all EDA methods.

### Dataset Stats

- **Source File**: `MachineLearningRating_v3.txt`
- **Key Columns**: `TotalPremium`, `TotalClaims`, `TransactionMonth`, `make`, `Province`, `ZipCode`, `CoverType`, `LossRatio` (derived).
- **Records**: [To be filled after preprocessing; e.g., ~100,000 rows after cleaning, depending on data].
- **Date Range**: `TransactionMonth` spans [start date] to [end date], formatted as `YYYY-MM-DD HH:MM:SS`.
- **Missing Values**: [To be filled after running `check_missing_values`].

---

## Task 2: Data Pipeline with DVC

### Methodology

- DVC Setup:

  - Installed DVC (pip install dvc) and added to requirements.txt.

  - Initialized DVC (dvc init) to enable data versioning.

  - Configured local remote storage at ~/dvc_storage (dvc remote add -d localstorage dvc_storage).

- Data Versioning:

  - Tracked data/MachineLearningRating_v3.txt and data/insurance_clean.csv with DVC (dvc add).

  - Committed .dvc files and data/.gitignore to Git for version control.

  - Created a new version of insurance_clean.csv by filtering to 2023 data, demonstrating DVC versioning.

  - Pushed data to local remote storage (dvc push) for reproducibility.

- Purpose: Ensures data inputs are version-controlled, enabling auditable and reproducible analyses for regulatory compliance in insurance.

---

## Task 3: Hypothesis Testing for Risk Drivers

### Methodology

- **Metrics:**

  - **Claim Frequency:** Proportion of policies with `TotalClaims > 0`.
  - **Claim Severity:** Average `TotalClaims` for policies with claims.
  - **Margin:** `TotalPremium - TotalClaims` per policy.

- **Hypotheses:**

  - H₀: No risk differences across provinces (Claim Frequency, Claim Severity).
  - H₀: No risk differences between zip codes (Claim Frequency, Claim Severity).
  - H₀: No significant margin difference between zip codes.
  - H₀: No significant risk difference between Women and Men (Claim Frequency, Claim Severity).

- **Data Segmentation:**

  - **Provinces:** Grouped by `Province` for all regions.
  - **Zip Codes:** Focused on top 5 zip codes by policy count to manage sample size.
  - **Gender:** Group A: Male, Group B: Female. Verified equivalence in `CoverType` and `Make` using chi-squared tests.

- **Statistical Testing (implemented in `src/insurance_eda.py`):**

  - **Claim Frequency:** Chi-squared test for categorical proportions.
  - **Claim Severity and Margin:**
    - Shapiro-Wilk test for normality.
    - If normal: ANOVA for multiple groups, t-test for gender.
    - If non-normal: Kruskal-Wallis for multiple groups, Mann-Whitney U for gender.
    - Tukey HSD post-hoc test for multiple comparisons when applicable.
  - **P-value Threshold:** Reject H₀ if p < 0.05.

- **Analysis and Reporting:**
  - Printed results with hypothesis, metric, statistical test, p-value, and decision.
  - Provided business recommendations for each rejected hypothesis (e.g., premium adjustments).
  - All tests executed via `insurance_eda.ipynb`, with DVC used for versioning data and results.

---

### Findings

#### Provinces (Claim Frequency)

- **Result:** Rejected H₀ (p < 0.0001, Chi-squared test). Significant differences in Claim Frequency across provinces.
- **Details:** Gauteng had the highest claim frequency (~0.34%), while Northern Cape had the lowest (~0.13%).
- **Business Recommendation:** Increase premiums in high-risk provinces like Gauteng. Offer competitive rates in low-risk areas like Northern Cape.

#### Provinces (Claim Severity)

- **Result:** Rejected H₀ (p < 0.0001, Kruskal-Wallis test). Significant differences in Claim Severity across provinces.
- **Details:** Tukey HSD showed no significant difference between some provinces, but others did differ.
- **Business Recommendation:** Use risk-based pricing where Claim Severity is higher; explore loss prevention in those regions.

#### Zip Codes (Claim Frequency)

- **Result:** _Pending — re-run `insurance_eda.ipynb` to update results._
- **Details:** [Placeholder for chi-squared results across top 5 zip codes]
- **Business Recommendation:** If significant, adjust underwriting and premiums by zip code risk profile.

#### Zip Codes (Claim Severity)

- **Result:** _Pending — re-run `insurance_eda.ipynb` to update results._
- **Details:** [Placeholder for Kruskal-Wallis/ANOVA results]
- **Business Recommendation:** Tailor pricing and coverage to reflect risk in high-severity zip codes.

#### Margin by Zip Codes

- **Result:** _Pending — re-run `insurance_eda.ipynby` to update results._
- **Details:** [Placeholder for margin analysis]
- **Business Recommendation:** Adjust pricing strategies in low-margin areas to protect profitability.

#### Gender (Claim Frequency)

- **Result:** _Pending — re-run `insurance_eda.ipynb` to update results._
- **Details:** [Placeholder for chi-squared results comparing male vs female claim frequency]
- **Business Recommendation:** If significant, consider risk-based pricing adjustments while adhering to regulatory standards.

#### Gender (Claim Severity)

- **Result:** Failed to reject H₀ (p = 0.2235, Mann-Whitney U test). No statistically significant difference.
- **Details:** Average Claim Severity — Male: 14,858.55; Female: 17,874.72.
- **Business Recommendation:** Maintain gender-neutral pricing for Claim Severity. Focus on more predictive variables.

---

## Task 4: Predictive Modeling for Risk-Based Pricing

### Methodology

#### Goals:

- **Claim Severity Prediction**: Predict `TotalClaims` for policies with `claims > 0` (regression).
- **Claim Probability Prediction**: Predict probability of a claim occurring (binary classification).
- **Premium Optimization**: Predict `TotalPremium` and develop a risk-based pricing model:
  - `Premium = (Predicted Claim Probability * Predicted Claim Severity) + Expense Loading + Profit Margin`.

#### Data Preparation (`src/data_preparation.py`):

- **Missing Data**: Imputed numerical columns with median, categorical with mode.
- **Feature Engineering**:
  - Created `VehicleAge` (proxy based on `TransactionMonth` year).
  - Added `HasClaim` (binary indicator for `claims > 0`).
  - Computed `PremiumPerClaim` (`TotalPremium / TotalClaims`).
  - Extracted `TransactionYear` and `TransactionMonthOfYear` from `TransactionMonth`.
- **Encoding**: Used `LabelEncoder` for categorical columns (`Province`, `ZipCode`, `CoverType`, `make`, `Gender`).
- **Train-Test Split**: 80:20 split for each task (severity, probability, premium).
- **Output**: Saved train/test datasets to `data/prepared/`.

#### Modeling:

##### Claim Severity (`src/model_claim_severity.py`):

- **Models**: Linear Regression, Random Forest, XGBoost.
- **Target**: `TotalClaims` (subset where `HasClaim == 1`).

##### Claim Probability (`src/model_claim_probability.py`):

- **Models**: Logistic Regression, Random Forest, XGBoost.
- **Target**: `HasClaim` (binary).

##### Premium Optimization (`src/model_premium.py`):

- **Models**: Linear Regression, Random Forest, XGBoost, Risk-Based Premium.
- **Risk-Based Premium**: Trained XGBoost on `(Claim Probability * Claim Severity) * (1 + 0.10 + 0.05)` (expense + profit).

#### Evaluation (`src/evaluate_models.py`):

- **Regression Metrics**: RMSE, R-squared.
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score.
- **SHAP Analysis**: Performed on XGBoost models to identify top 5-10 features.
- **Output**: Saved metrics to `results/evaluation_metrics.csv` and SHAP plots to `results/`.

#### Git Workflow:

- Merged `task-3` into `main` via Pull Request.
- Created `task-4` branch for modeling work.
- Committed changes with descriptive messages.

#### DVC:

- Tracked model files (`models/*.pkl`) and evaluation results (`results/evaluation_metrics.csv`).

### Evaluation Results

#### Claim Severity:

- **[Pending]**: Run `evaluate_models.py` to populate RMSE and R² for Linear Regression, Random Forest, XGBoost.
- **SHAP Insights**: _Example_: "SHAP analysis reveals that for every year older a vehicle is, the predicted claim amount increases by X Rand, holding other factors constant. This supports age-based premium adjustments."

#### Claim Probability:

- **[Pending]**: Run `evaluate_models.py` to populate Accuracy, Precision, Recall, F1.
- **SHAP Insights**: _Example_: "ZipCode_encoded is the top feature, indicating geographic risk drives claim likelihood. High-risk areas like Gauteng justify higher premiums."

#### Premium Optimization:

- **[Pending]**: Run `evaluate_models.py` to populate RMSE and R².
- **SHAP Insights**: _Example_: "LossRatio is a key driver, suggesting policies with high historical claims require premium increases to maintain profitability."

### Business Recommendations:

- Use XGBoost models for deployment due to superior performance.
- Adjust premiums based on SHAP-identified features (e.g., vehicle age, geographic risk).
- Implement risk-based pricing to attract low-risk clients with competitive rates.

---

## File Structure

project/
├── data/
│ ├── insurance_clean.csv
│ ├── MachineLearningRating_v3.txt
| ├── insurance_clean.csv.dvc
│ ├── MachineLearningRating_v3.txt.dvc
| ├── .gitignore
|
├── src/
│ ├── insurance_eda.py
│ └── preprocessing.py
├── notebook/
│ ├── insurance_eda.ipynb
├── requirements.txt
├── .gitignore
└── README.md

---

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt

   ```

2. Run each step using the Jupyter notebooks in the notebook/ folder.

3. Set Up DVC:

```bash
dvc init
dvc remote add -d localstorage ~/dvc_storage
dvc pull

```

---

## KPIs

- Proactivity to self-learn - sharing references.

- EDA techniques to understand data and discover insights,

- Demonstrating Stats understanding by using suitable statistical distributions and plots to provide evidence for actionable insights gained from EDA.

- Reproducible data pipeline with DVC for auditability and compliance.

- Statistical validation of risk drivers for segmentation strategy.

- Predictive modeling with robust evaluation and interpretability for business impact.

---
