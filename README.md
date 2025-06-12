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

## File Structure

project/
├── data/
│ ├── insurance_clean.csv
│ ├── MachineLearningRating_v3.txt
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

---

## KPIs

- Proactivity to self-learn - sharing references.

- EDA techniques to understand data and discover insights,

- Demonstrating Stats understanding by using suitable statistical distributions and plots to provide evidence for actionable insights gained from EDA.

---
