import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("SOFTWARE DEFECT DATASET PREPROCESSING PIPELINE")

# STEP 1: LOAD AND EXPLORE DATA

print("STEP 1: DATA LOADING AND EXPLORATION")

# Load the dataset
data_path = "SoftwareDefectDataset.csv"
df = pd.read_csv(data_path)

print(f"\nFirst 5 Rows:")
print(df.head())


# STEP 2: PREPROCESSING

print("STEP 2: PREPROCESSING")

# 2.1 Check for missing values
print("\n2.1 Checking Missing Values:")
missing_values = df.isnull().sum()
total_missing = missing_values.sum()
print(f"Total Missing Values: {total_missing}")
if total_missing > 0:
    print("Missing values detected:")
    print(missing_values[missing_values > 0])
else:
    print("No missing values detected!")

# 2.2 Check for duplicate rows
print("\n2.2 Checking Duplicate Rows:")
duplicates = df.duplicated().sum()
print(f"Duplicate Rows: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()
    print(f" Removed {duplicates} duplicate rows. New shape: {df.shape}")
else:
    print(" No duplicate rows found!")

# 2.3 Check data distribution
print("\n2.3 Data Distribution Check:")
print("Value counts for target variable (DEFECT_LABEL):")
print(df['DEFECT_LABEL'].value_counts())
print(f"\nClass Distribution:")
print(f"  - Non-defective (0): {(df['DEFECT_LABEL'] == 0).sum()} ({(df['DEFECT_LABEL'] == 0).mean()*100:.2f}%)")
print(f"  - Defective (1): {(df['DEFECT_LABEL'] == 1).sum()} ({(df['DEFECT_LABEL'] == 1).mean()*100:.2f}%)")

# STEP 3: OUTLIER DETECTION AND HANDLING

print("STEP 3: OUTLIER DETECTION AND HANDLING")

# Function to detect outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Function to detect outliers using Z-score method
def detect_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = data[z_scores > threshold]
    return outliers, z_scores

# Get feature columns (exclude target)
feature_columns = df.columns[:-1].tolist()
target_column = 'DEFECT_LABEL'

print("\n3.1 Outlier Detection using IQR Method:")

outlier_summary = {}
for col in feature_columns:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    outlier_count = len(outliers)
    outlier_summary[col] = outlier_count
    if outlier_count > 0:
        print(f"  {col}: {outlier_count} outliers (range: {lower:.4f} - {upper:.4f})")

total_outliers = sum(outlier_summary.values())
print(f"\nTotal outliers detected: {total_outliers}")

# 3.2 Outlier Handling Options
print("\n3.2 Outlier Handling:")
print("  1. Remove outliers")

# Create a copy for handling
df_processed = df.copy()

# Apply winsorization to handle outliers
for col in feature_columns:
    lower_percentile = df_processed[col].quantile(0.05)
    upper_percentile = df_processed[col].quantile(0.95)
    df_processed[col] = np.clip(df_processed[col], lower_percentile, upper_percentile)

print(" Outliers handled using winsorization method")

# Show comparison
print("\n3.3 Before vs After Outlier Handling:")
print("-" * 50)
comparison = pd.DataFrame({
    'Before': df[feature_columns].std(),
    'After': df_processed[feature_columns].std()
})
print(comparison)

# STEP 4: CORRELATION ANALYSIS

print("STEP 4: CORRELATION ANALYSIS")

# Calculate correlation matrix
correlation_matrix = df_processed.corr()

print("\n4.1 Correlation with Target Variable (DEFECT_LABEL):")
target_correlations = correlation_matrix['DEFECT_LABEL'].drop('DEFECT_LABEL').sort_values(ascending=False)
print(target_correlations)

# Identify highly correlated features
print("\n4.2 Highly Correlated Feature Pairs (|r| > 0.7):")
for i in range(len(feature_columns)):
    for j in range(i+1, len(feature_columns)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.7:
            print(f"  {feature_columns[i]} ↔ {feature_columns[j]}: {corr:.4f}")

# STEP 5: FEATURE SELECTION

print("STEP 5: FEATURE SELECTION")

# 5.1 Correlation-based feature selection
print("\n5.1 Correlation-Based Feature Selection:")
print("Features ranked by correlation with target:")
ranked_features = target_correlations.abs().sort_values(ascending=False)
for i, (feature, corr) in enumerate(ranked_features.items(), 1):
    direction = "+" if target_correlations[feature] > 0 else "-"
    print(f"  {i}. {feature}: {target_correlations[feature]:.4f} ({direction})")

# 5.2 Select features with significant correlation (|r| > 0.1)
significant_features = ranked_features[ranked_features > 0.1].index.tolist()
print(f"\n5.2 Selected Features (|correlation| > 0.1):")
print(f"  {significant_features}")
print(f"  Total selected: {len(significant_features)} out of {len(feature_columns)}")

# 5.3 Feature importance using statistical tests
print("\n5.3 Statistical Feature Importance:")

# Point-biserial correlation for binary target
from scipy.stats import pointbiserialr

feature_importance = {}
for col in feature_columns:
    corr, p_value = pointbiserialr(df_processed[col], df_processed[target_column])
    feature_importance[col] = {'correlation': corr, 'p_value': p_value}

importance_df = pd.DataFrame(feature_importance).T
importance_df = importance_df.sort_values('p_value')
print(importance_df)

# STEP 6: VISUALIZATION


print("STEP 6: VISUALIZATION")

# Create multiple subplots for comprehensive visualization
fig = plt.figure(figsize=(20, 30))

# 6.1 Data Distribution Histograms
print("\n6.1 Creating Distribution Plots...")
ax1 = fig.add_subplot(4, 2, 1)
df_processed.hist(bins=30, ax=ax1, figsize=(16, 20))
ax1.set_title('Feature Distributions (Histograms)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('distribution_histograms.png', dpi=150, bbox_inches='tight')
print("Saved: distribution_histograms.png")

# 6.2 Correlation Heatmap
ax2 = fig.add_subplot(4, 2, 2)
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
            fmt='.2f', linewidths=0.5, ax=ax2)
ax2.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
print("Saved: correlation_heatmap.png")

# 6.3 Box Plots for Outlier Detection
ax3 = fig.add_subplot(4, 2, 3)
df_processed.boxplot(ax=ax3, vert=True, figsize=(16, 10))
ax3.set_title('Box Plots (Outlier Visualization)', fontsize=14, fontweight='bold')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('boxplots_outliers.png', dpi=150, bbox_inches='tight')
print("Saved: boxplots_outliers.png")

# 6.4 Target Correlation Bar Chart
ax4 = fig.add_subplot(4, 2, 4)
colors = ['green' if x > 0 else 'red' for x in target_correlations]
target_correlations.plot(kind='barh', ax=ax4, color=colors)
ax4.set_title('Feature Correlation with DEFECT_LABEL', fontsize=14, fontweight='bold')
ax4.set_xlabel('Correlation Coefficient')
ax4.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('target_correlation.png', dpi=150, bbox_inches='tight')
print("Saved: target_correlation.png")

# 6.5 Pairplot for top features (using figure-level function)
top_features = ['LOC', 'CYCLO', 'DIFFICULTY', 'DEFECT_LABEL']
pairfig = sns.pairplot(df_processed[top_features], hue='DEFECT_LABEL', diag_kind='kde')
pairfig.fig.suptitle('Pairplot for Top Features', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
pairfig.savefig('pairplot_top_features.png', dpi=150, bbox_inches='tight')
print("Saved: pairplot_top_features.png")

# 6.6 Feature Importance Plot
ax6 = fig.add_subplot(4, 2, 6)
importance_df_sorted = importance_df.sort_values('p_value')
colors = ['green' if p < 0.05 else 'red' for p in importance_df_sorted['p_value']]
ax6.barh(importance_df_sorted.index, -np.log10(importance_df_sorted['p_value']), color=colors)
ax6.set_xlabel('-log10(p-value)')
ax6.set_title('Feature Importance (Statistical Significance)', fontsize=14, fontweight='bold')
ax6.axvline(x=-np.log10(0.05), color='black', linestyle='--', label='p=0.05')
ax6.legend()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("Saved: feature_importance.png")

# 6.7 Class Distribution
ax7 = fig.add_subplot(4, 2, 7)
class_counts = df_processed['DEFECT_LABEL'].value_counts()
ax7.pie(class_counts, labels=['Non-defective', 'Defective'], autopct='%1.1f%%',
        colors=['lightgreen', 'salmon'], explode=(0.05, 0.05))
ax7.set_title('Class Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: class_distribution.png")

# 6.8 Correlation with Target - Sorted
ax8 = fig.add_subplot(4, 2, 8)
sorted_corr = target_correlations.sort_values()
colors = ['red' if x < 0 else 'blue' for x in sorted_corr]
sorted_corr.plot(kind='barh', ax=ax8, color=colors)
ax8.set_title('Sorted Correlations with DEFECT_LABEL', fontsize=14, fontweight='bold')
ax8.set_xlabel('Correlation')
ax8.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('sorted_correlations.png', dpi=150, bbox_inches='tight')
print("Saved: sorted_correlations.png")

plt.savefig('all_visualizations.png', dpi=150, bbox_inches='tight')
print("\n Saved: all_visualizations.png")