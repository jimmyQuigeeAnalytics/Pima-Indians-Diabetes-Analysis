"""
Simple EDA script for Pima Indians Diabetes Dataset
"""
import pandas as pd
import numpy as np
from scipy import stats

print("="*80)
print("PIMA INDIANS DIABETES - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load data
df = pd.read_csv('diabetes.csv')
print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumns: {', '.join(df.columns.tolist())}")

# Basic stats
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)
print(df.describe())

# Outcome distribution
print("\n" + "="*80)
print("OUTCOME DISTRIBUTION")
print("="*80)
outcome_counts = df['Outcome'].value_counts()
outcome_pct = df['Outcome'].value_counts(normalize=True) * 100
print(f"Non-Diabetic (0): {outcome_counts[0]} ({outcome_pct[0]:.2f}%)")
if 1 in outcome_counts:
    print(f"Diabetic (1): {outcome_counts[1]} ({outcome_pct[1]:.2f}%)")

# Missing values
print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values found!")
else:
    print(missing[missing > 0])

# Zero values check
print("\n" + "="*80)
print("BIOLOGICALLY IMPLAUSIBLE ZEROS")
print("="*80)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    zeros = (df[col] == 0).sum()
    print(f"{col}: {zeros} zeros ({(zeros/len(df)*100):.2f}%)")

# Statistics by outcome
print("\n" + "="*80)
print("STATISTICS BY OUTCOME")
print("="*80)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

print("\nNon-Diabetic (Outcome = 0):")
print(df[df['Outcome'] == 0][features].describe())

print("\nDiabetic (Outcome = 1):")
print(df[df['Outcome'] == 1][features].describe())

# Correlation
print("\n" + "="*80)
print("CORRELATION WITH OUTCOME")
print("="*80)
corr = df.corr()['Outcome'].sort_values(ascending=False)
corr = corr[corr.index != 'Outcome']
for feature, val in corr.items():
    print(f"{feature}: {val:.3f}")

# Statistical tests
print("\n" + "="*80)
print("STATISTICAL TESTS (T-tests)")
print("="*80)
print(f"{'Feature':<25} {'p-value':<15} {'Significant'}")
print("-"*60)
for feature in features:
    non_diab = df[df['Outcome'] == 0][feature]
    diab = df[df['Outcome'] == 1][feature]
    
    if feature in zero_cols:
        non_diab = non_diab[non_diab > 0]
        diab = diab[diab > 0]
    
    if len(non_diab) > 0 and len(diab) > 0:
        _, p_val = stats.ttest_ind(non_diab, diab)
        sig = "Yes" if p_val < 0.05 else "No"
        print(f"{feature:<25} {p_val:>10.4f}     {sig}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

