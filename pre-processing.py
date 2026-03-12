import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset
df = pd.read_csv('SoftwareDefectDataset.csv')

print(f"\nDataset Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows of the dataset:")
print(df.head())

# 2a: Check for missing values
missing_values = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

print("\nSum of missing values in each column:")
print(missing_values)

print("\nPercentage of missing values in each column:")
print(missing_percent.round(2))

print(f"\nTotal missing values in dataset: {df.isnull().sum().sum()}")

# 2b: Show missing values heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Columns', fontsize=12)
plt.ylabel('Rows', fontsize=12)
plt.tight_layout()
plt.savefig('missing_values_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Check if there are missing values before filling
if df.isnull().sum().sum() > 0:
    # Fill missing values with mean for each numeric column
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
            print(f"Filled column '{column}' with mean value: {mean_value:.4f}")
    print("\nMissing values after filling:")
    print(df.isnull().sum())
else:
    print("No missing values found in the dataset!")

# 3a: Remove duplicate rows
duplicates_before = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates_before}")

if duplicates_before > 0:
    df = df.drop_duplicates()
    print(f"Removed {duplicates_before} duplicate rows")
else:
    print("No duplicate rows found")

# 3b: Check for zero variance columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
variances = df[numeric_cols].var()
zero_var_cols = variances[variances == 0].index.tolist()

if zero_var_cols:
    print(f"Zero variance columns found: {zero_var_cols}")
    print("Dropping zero variance columns...")
    df = df.drop(columns=zero_var_cols)
else:
    print("No zero variance columns found")

# 3c: Handle outliers using IQR method
# Separate features and target
target_col = 'DEFECT_LABEL'
feature_cols = [col for col in df.columns if col != target_col]

# Count outliers before
outlier_counts = {}
for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    outlier_counts[col] = outliers

print("Outliers detected per column:")
for col, count in outlier_counts.items():
    print(f"  {col}: {count} outliers")

# Cap outliers instead of removing (Winsorization)
for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)


# Reload original data to demonstrate feature engineering
df_original = pd.read_csv('SoftwareDefectDataset.csv')
# Fill missing values in original for consistency
for column in df_original.columns:
    if df_original[column].isnull().sum() > 0:
        df_original[column].fillna(df_original[column].mean(), inplace=True)

df_engineered = df_original.copy()

# Create ratio features
df_engineered['LOC_per_CYCLO'] = df_engineered['LOC'] / (df_engineered['CYCLO'] + 1)
df_engineered['VOLUME_per_LENGTH'] = df_engineered['VOLUME'] / (df_engineered['LENGTH'] + 1)
df_engineered['OPERATORS_OPERANDS_RATIO'] = df_engineered['NUM_OPERATORS'] / (df_engineered['NUM_OPERANDS'] + 1)
df_engineered['FAN_IN_OUT_RATIO'] = df_engineered['INT_FAN_IN'] / (df_engineered['INT_FAN_OUT'] + 1)
df_engineered['DIFFICULTY_VOLUME'] = df_engineered['DIFFICULTY'] * df_engineered['VOLUME']

print("Created interaction features:")
print("  - LOC_per_CYCLO")
print("  - VOLUME_per_LENGTH")
print("  - OPERATORS_OPERANDS_RATIO")
print("  - FAN_IN_OUT_RATIO")
print("  - DIFFICULTY_VOLUME")

# 4b: Create polynomial features (squared terms)
print("\n--- 4b: Creating Polynomial Features ---")
df_engineered['LOC_squared'] = df_engineered['LOC'] ** 2
df_engineered['CYCLO_squared'] = df_engineered['CYCLO'] ** 2
df_engineered['VOLUME_squared'] = df_engineered['VOLUME'] ** 2

print("Created polynomial features:")
print("  - LOC_squared")
print("  - CYCLO_squared")
print("  - VOLUME_squared")

# 4c: Log transformation for skewed features
print("\n--- 4c: Log Transformation ---")
for col in ['LOC', 'VOLUME', 'LENGTH']:
    df_engineered[f'{col}_log'] = np.log1p(df_engineered[col])
    print(f"  Created {col}_log (log1p transformation)")

print(f"\nEngineered dataset shape: {df_engineered.shape}")
print(f"New total columns: {len(df_engineered.columns)}")

# Use the engineered dataframe for further processing
df = df_engineered

# Separate features and target
target_col = 'DEFECT_LABEL'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# 5a: StandardScaler (z-score normalization)
print("\n--- 5a: StandardScaler (Z-score Normalization) ---")
scaler_standard = StandardScaler()
X_standardized = scaler_standard.fit_transform(X)
X_standardized = pd.DataFrame(X_standardized, columns=X.columns)

print("Standardization applied (mean=0, std=1)")
print("\nFirst 5 rows after standardization:")
print(X_standardized.head())

# 5b: Show statistics after scaling
print("\n--- 5b: Statistics After Scaling ---")
print("\nMean of scaled features (should be ~0):")
print(X_standardized.mean().round(6))
print("\nStandard deviation of scaled features (should be ~1):")
print(X_standardized.std().round(6))

# Save preprocessed data
X_standardized.to_csv('preprocessed_features.csv', index=False)
y.to_csv('preprocessed_target.csv', index=False)


