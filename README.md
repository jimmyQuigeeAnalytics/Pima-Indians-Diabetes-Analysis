# Pima Indians Diabetes - Exploratory Data Analysis

## Overview

This project contains a comprehensive exploratory data analysis (EDA) of the Pima Indians Diabetes dataset. The analysis examines various aspects of diabetes in Pima Indian women, focusing on identifying patterns, relationships, and risk factors associated with diabetes.

## Context

Diabetes is one of the most frequent diseases worldwide, and the number of diabetic patients continues to grow. Research was conducted on the Pima Indian tribe, where it was found that women are prone to diabetes at an early age. This dataset contains information about female patients at least 21 years old of Pima Indian heritage.

## Dataset Information

- **Total Samples**: 768
- **Features**: 8
- **Target Variable**: Outcome (0 = Non-Diabetic, 1 = Diabetic)

### Features

1. **Pregnancies**: Number of times pregnant
2. **Glucose**: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg/(height in m)²)
7. **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history
8. **Age**: Age in years

## Files in This Project

1. **`diabetes.csv`**: The dataset file
2. **`Diabetes_EDA.ipynb`**: Comprehensive Jupyter notebook with complete EDA analysis
3. **`diabetes_analysis.py`**: Python script version of the analysis (alternative to notebook)
4. **`eda_simple.py`**: Simplified version for quick analysis

## Analysis Components

The comprehensive analysis includes:

### 1. Data Loading and Inspection
- Dataset shape and structure
- Column information
- Data types
- First few rows preview

### 2. Data Quality Assessment
- Missing values check
- Identification of biologically implausible zeros
- Data quality issues and recommendations

### 3. Statistical Summary
- Descriptive statistics for all features
- Statistics grouped by outcome (diabetic vs non-diabetic)
- Outcome distribution analysis

### 4. Visualizations
- Outcome distribution bar chart
- Feature distribution histograms
- Box plots comparing features by outcome
- Correlation heatmap
- Violin plots for key features
- Mean comparison bar charts
- Pair plots for key features

### 5. Statistical Significance Tests
- T-tests comparing means between diabetic and non-diabetic groups
- Identification of statistically significant features

### 6. Key Insights and Summary
- Dataset overview
- Data quality findings
- Key statistical differences
- Top risk factors
- Conclusions and recommendations

## Key Findings

### Data Quality
- No explicit missing values (NaN) in the dataset
- However, many zero values exist in features where zero doesn't make biological sense:
  - Glucose: ~0.65% zeros
  - BloodPressure: ~4.56% zeros
  - SkinThickness: ~29.56% zeros
  - Insulin: ~48.70% zeros
  - BMI: ~1.43% zeros
- These zeros likely represent missing data rather than actual measurements

### Outcome Distribution
- The dataset is relatively balanced between diabetic and non-diabetic cases
- Approximately 65% non-diabetic and 35% diabetic cases

### Key Risk Factors
Based on correlation analysis, the top risk factors for diabetes (in order of correlation magnitude) are:

1. **Glucose**: Strongest positive correlation with diabetes
2. **BMI**: Higher BMI associated with increased diabetes risk
3. **Age**: Older age associated with higher diabetes risk
4. **DiabetesPedigreeFunction**: Family history score positively correlated with diabetes
5. **Pregnancies**: Number of pregnancies shows positive correlation

### Statistical Differences
Features showing statistically significant differences (p < 0.05) between diabetic and non-diabetic groups:
- Glucose (higher in diabetic group)
- BMI (higher in diabetic group)
- Age (higher in diabetic group)
- DiabetesPedigreeFunction (higher in diabetic group)
- Pregnancies (higher in diabetic group)

## How to Use

### Option 1: Jupyter Notebook (Recommended)
1. Open `Diabetes_EDA.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially
3. The notebook will generate all visualizations and statistical analyses

### Option 2: Python Script
1. Run the Python script:
   ```bash
   python diabetes_analysis.py
   ```
2. The script will generate visualizations as PNG files and print analysis results to the console

### Option 3: Simple Analysis
For a quick analysis without visualizations:
```bash
python eda_simple.py
```

## Requirements

The analysis requires the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scipy

Install requirements:
```bash
pip install pandas numpy matplotlib seaborn scipy
```

## Recommendations for Further Analysis

1. **Data Preprocessing**: Handle zero values appropriately (treat as missing data or impute)
2. **Feature Engineering**: Create new features or transform existing ones
3. **Machine Learning**: Build predictive models (classification) to predict diabetes
4. **Advanced Visualizations**: Create more detailed visualizations for specific relationships
5. **Statistical Modeling**: Perform regression analysis or other advanced statistical tests
6. **Cross-Validation**: Validate findings using cross-validation techniques

## Notes

- The dataset contains some data quality issues that should be addressed before building predictive models
- Zero values in certain features likely represent missing data and should be handled appropriately
- The analysis focuses on exploratory analysis; predictive modeling would be the next step

## Author

Analysis conducted as part of the Applied Data Science Program.

## License

This analysis is for educational purposes.

