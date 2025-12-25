"""
Comprehensive Exploratory Data Analysis of Pima Indians Diabetes Dataset
=======================================================================
This script performs a thorough analysis of diabetes in Pima Indian women,
including data quality checks, statistical summaries, and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import sys
warnings.filterwarnings('ignore')

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        try:
            plt.style.use('ggplot')
        except:
            pass

try:
    sns.set_palette("husl")
except:
    pass

# Load the dataset
print("="*80)
print("PIMA INDIANS DIABETES - EXPLORATORY DATA ANALYSIS")
print("="*80)
print("\n1. LOADING AND INSPECTING DATA")
print("-"*80)

try:
    df = pd.read_csv('diabetes.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nFirst few rows:")
print(df.head(10))

print(f"\nData Types:")
print(df.dtypes)

print(f"\nBasic Information:")
df.info()

# ============================================================================
# 2. DATA QUALITY CHECKS
# ============================================================================
print("\n\n2. DATA QUALITY ASSESSMENT")
print("-"*80)

print("\nMissing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
missing_nonzero = missing_df[missing_df['Missing Count'] > 0]
if len(missing_nonzero) > 0:
    print(missing_nonzero)
else:
    print("  No missing values found!")

# Check for zeros in columns where zero doesn't make biological sense
print("\n\nChecking for biologically implausible zeros:")
zero_check_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_counts = {}
for col in zero_check_cols:
    zero_count = (df[col] == 0).sum()
    zero_pct = (zero_count / len(df)) * 100
    zero_counts[col] = {'count': zero_count, 'percentage': zero_pct}
    print(f"  {col}: {zero_count} zeros ({zero_pct:.2f}%)")

# ============================================================================
# 3. STATISTICAL SUMMARIES
# ============================================================================
print("\n\n3. STATISTICAL SUMMARY")
print("-"*80)

print("\nDescriptive Statistics for All Features:")
print(df.describe())

print("\n\nDescriptive Statistics by Outcome (Diabetes Status):")
print("\nNon-Diabetic (Outcome = 0):")
print(df[df['Outcome'] == 0].describe())

print("\n\nDiabetic (Outcome = 1):")
print(df[df['Outcome'] == 1].describe())

# Outcome distribution
print("\n\nOutcome Distribution:")
outcome_counts = df['Outcome'].value_counts()
outcome_pct = df['Outcome'].value_counts(normalize=True) * 100
print(f"  Non-Diabetic (0): {outcome_counts[0]} ({outcome_pct[0]:.2f}%)")
if 1 in outcome_counts:
    print(f"  Diabetic (1): {outcome_counts[1]} ({outcome_pct[1]:.2f}%)")

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
print("\n\n4. GENERATING VISUALIZATIONS")
print("-"*80)

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

try:
    # 4.1 Outcome Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    outcome_counts.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
    ax.set_title('Distribution of Diabetes Outcome', fontsize=14, fontweight='bold')
    ax.set_xlabel('Outcome (0=Non-Diabetic, 1=Diabetic)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticklabels(['Non-Diabetic', 'Diabetic'], rotation=0)
    for i, v in enumerate(outcome_counts):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('1_outcome_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 1_outcome_distribution.png")
except Exception as e:
    print(f"Error creating outcome distribution plot: {e}")

try:
    # 4.2 Feature Distributions - Histograms
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        df[feature].hist(bins=30, ax=axes[idx], edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].axvline(df[feature].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df[feature].mean():.2f}')
        axes[idx].legend()
    
    # Remove the last empty subplot
    fig.delaxes(axes[8])
    plt.tight_layout()
    plt.savefig('2_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 2_feature_distributions.png")
except Exception as e:
    print(f"Error creating feature distributions plot: {e}")

try:
    # 4.3 Box Plots by Outcome
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, feature in enumerate(features):
        data_0 = df[df['Outcome'] == 0][feature]
        data_1 = df[df['Outcome'] == 1][feature]
        bp = axes[idx].boxplot([data_0, data_1], labels=['Non-Diabetic', 'Diabetic'], patch_artist=True)
        bp['boxes'][0].set_facecolor('skyblue')
        bp['boxes'][1].set_facecolor('salmon')
        axes[idx].set_title(f'{feature} by Outcome', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(feature, fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions by Diabetes Outcome', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('3_boxplots_by_outcome.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 3_boxplots_by_outcome.png")
except Exception as e:
    print(f"Error creating boxplots: {e}")

try:
    # 4.4 Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix of Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 4_correlation_heatmap.png")
except Exception as e:
    print(f"Error creating correlation heatmap: {e}")

try:
    # 4.5 Violin Plots - Key Features by Outcome
    key_features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, feature in enumerate(key_features):
        data_0 = df[df['Outcome'] == 0][feature]
        data_1 = df[df['Outcome'] == 1][feature]
        data_to_plot = [data_0, data_1]
        parts = axes[idx].violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_xticklabels(['Non-Diabetic', 'Diabetic'])
        axes[idx].set_ylabel(feature, fontsize=12)
        axes[idx].set_title(f'{feature} Distribution by Outcome', fontsize=13, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Key Features: Distribution Comparison by Outcome', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('5_violin_plots_key_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 5_violin_plots_key_features.png")
except Exception as e:
    print(f"Error creating violin plots: {e}")

try:
    # 4.6 Pair Plot for Key Features
    key_features_pair = ['Glucose', 'BMI', 'Age', 'Outcome']
    g = sns.pairplot(df[key_features_pair], hue='Outcome', diag_kind='kde', 
                     palette={0: 'skyblue', 1: 'salmon'}, plot_kws={'alpha': 0.6})
    g.fig.suptitle('Pair Plot of Key Features by Outcome', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('6_pairplot_key_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 6_pairplot_key_features.png")
except Exception as e:
    print(f"Error creating pairplot: {e}")

try:
    # 4.7 Mean Comparison by Outcome
    fig, ax = plt.subplots(figsize=(14, 8))
    means_by_outcome = df.groupby('Outcome')[features].mean()
    x = np.arange(len(features))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, means_by_outcome.loc[0], width, 
                   label='Non-Diabetic', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, means_by_outcome.loc[1], width, 
                   label='Diabetic', color='salmon', alpha=0.8)
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    ax.set_title('Mean Feature Values by Diabetes Outcome', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('7_mean_comparison_by_outcome.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: 7_mean_comparison_by_outcome.png")
except Exception as e:
    print(f"Error creating mean comparison plot: {e}")

# ============================================================================
# 5. STATISTICAL TESTS
# ============================================================================
print("\n\n5. STATISTICAL SIGNIFICANCE TESTS")
print("-"*80)

print("\nT-tests comparing means between Diabetic and Non-Diabetic groups:")
print("(H0: No difference in means, H1: Means are different)")
print("\n" + "-"*60)
print(f"{'Feature':<25} {'t-statistic':<15} {'p-value':<15} {'Significant'}")
print("-"*60)

significant_features = []
for feature in features:
    try:
        non_diabetic = df[df['Outcome'] == 0][feature]
        diabetic = df[df['Outcome'] == 1][feature]
        
        # Remove zeros for features where zero doesn't make sense
        if feature in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            non_diabetic = non_diabetic[non_diabetic > 0]
            diabetic = diabetic[diabetic > 0]
        
        if len(non_diabetic) > 0 and len(diabetic) > 0:
            t_stat, p_value = stats.ttest_ind(non_diabetic, diabetic)
            is_significant = "Yes" if p_value < 0.05 else "No"
            
            if p_value < 0.05:
                significant_features.append(feature)
            
            print(f"{feature:<25} {t_stat:>10.4f}     {p_value:>10.4f}     {is_significant}")
    except Exception as e:
        print(f"{feature:<25} Error: {e}")

print("-"*60)
print(f"\nFeatures with statistically significant differences (p < 0.05): {len(significant_features)}")
if significant_features:
    print(f"Significant features: {', '.join(significant_features)}")

# ============================================================================
# 6. KEY INSIGHTS AND SUMMARY
# ============================================================================
print("\n\n6. KEY INSIGHTS AND SUMMARY")
print("="*80)

print("\n📊 DATASET OVERVIEW:")
print(f"  • Total samples: {len(df)}")
print(f"  • Features: {len(features)}")
print(f"  • Diabetic cases: {outcome_counts[1] if 1 in outcome_counts else 0} ({outcome_pct[1]:.1f}% if 1 in outcome_pct else 0)")
print(f"  • Non-diabetic cases: {outcome_counts[0]} ({outcome_pct[0]:.1f}%)")

print("\n🔍 DATA QUALITY FINDINGS:")
zero_total = sum([zero_counts[col]['count'] for col in zero_check_cols])
print(f"  • Total biologically implausible zeros: {zero_total}")
for col in zero_check_cols:
    if zero_counts[col]['count'] > 0:
        print(f"    - {col}: {zero_counts[col]['count']} zeros ({zero_counts[col]['percentage']:.1f}%)")

print("\n📈 KEY STATISTICAL DIFFERENCES (Diabetic vs Non-Diabetic):")
for feature in significant_features:
    mean_0 = df[df['Outcome'] == 0][feature].mean()
    mean_1 = df[df['Outcome'] == 1][feature].mean()
    diff = mean_1 - mean_0
    diff_pct = (diff / mean_0) * 100 if mean_0 != 0 else 0
    print(f"  • {feature}:")
    print(f"    - Non-Diabetic mean: {mean_0:.2f}")
    print(f"    - Diabetic mean: {mean_1:.2f}")
    print(f"    - Difference: {diff:+.2f} ({diff_pct:+.1f}%)")

print("\n💡 CORRELATION WITH OUTCOME:")
try:
    outcome_corr = df.corr()['Outcome'].sort_values(ascending=False)
    outcome_corr = outcome_corr[outcome_corr.index != 'Outcome']
    print("  Features most correlated with diabetes outcome:")
    for feature, corr in outcome_corr.items():
        direction = "↑" if corr > 0 else "↓"
        print(f"    {direction} {feature}: {corr:.3f}")
    
    print("\n🎯 TOP RISK FACTORS (by correlation magnitude):")
    top_risk = outcome_corr.abs().sort_values(ascending=False).head(5)
    for i, (feature, corr) in enumerate(top_risk.items(), 1):
        print(f"  {i}. {feature} (correlation: {outcome_corr[feature]:.3f})")
except Exception as e:
    print(f"  Error calculating correlations: {e}")

print("\n" + "="*80)
print("Analysis complete! All visualizations have been saved.")
print("="*80)
