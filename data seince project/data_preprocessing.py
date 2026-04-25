import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

df_original = pd.read_csv('SoftwareDefectDataset.csv')
print(f"\nDataset Loaded: {df_original.shape}")
print(df_original.head())

# STEP 1: MISSING VALUES

print("STEP 1: MISSING VALUES HANDLING")

df_missing = df_original.copy()
np.random.seed(42)
columns_to_affect = ['LOC', 'CYCLO', 'VOLUME', 'DIFFICULTY', 'NUM_OPERATORS']

for col in columns_to_affect:
    missing_indices = np.random.choice(df_missing.index, size=int(len(df_missing) * 0.06), replace=False)
    df_missing.loc[missing_indices, col] = np.nan

print(f"\nMissing Values Introduced:\n{df_missing.isnull().sum()[df_missing.isnull().sum() > 0]}")

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df_missing.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=ax)
ax.set_title('Missing Values BEFORE Handling', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_missing_values_before.png', dpi=150, bbox_inches='tight')
plt.show()

df_handled = df_missing.copy()
for col in columns_to_affect:
    df_handled[col].fillna(df_handled[col].median(), inplace=True)

print(f"\nMissing Values After Handling:\n{df_handled.isnull().sum()}")

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df_handled.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=ax)
ax.set_title('Missing Values AFTER Handling', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('02_missing_values_after.png', dpi=150, bbox_inches='tight')
plt.show()


# STEP 2: OUTLIERS
print("STEP 2: OUTLIERS HANDLING")

df_outliers = df_handled.copy()
outlier_columns = ['LOC', 'VOLUME', 'BRANCH_COUNT']
np.random.seed(123)

for col in outlier_columns:
    num_outliers = np.random.randint(3, 6)
    outlier_indices = np.random.choice(df_outliers.index, size=num_outliers, replace=False)
    max_val = df_outliers[col].max()
    min_val = df_outliers[col].min()
    for idx in outlier_indices:
        if np.random.rand() > 0.5:
            df_outliers.loc[idx, col] = max_val * np.random.uniform(5, 10)
        else:
            df_outliers.loc[idx, col] = min_val * np.random.uniform(-5, -2)

print(f"\nOutliers introduced in: {outlier_columns}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(outlier_columns):
    axes[i].boxplot(df_outliers[col].dropna(), vert=True)
    axes[i].set_title(f'{col} BEFORE Handling', fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
plt.suptitle('Outliers BEFORE Handling', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_outliers_before.png', dpi=150, bbox_inches='tight')
plt.show()

df_clean = df_outliers.copy()
for col in df_clean.columns[:-1]:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(outlier_columns):
    axes[i].boxplot(df_clean[col].dropna(), vert=True)
    axes[i].set_title(f'{col} AFTER Handling', fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
plt.suptitle('Outliers AFTER Handling', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('04_outliers_after.png', dpi=150, bbox_inches='tight')
plt.show()

# STEP 3: CORRELATION

print("STEP 3: CORRELATION ANALYSIS")

correlation_matrix = df_clean.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, mask=mask, ax=ax)
ax.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

target_corr = correlation_matrix['DEFECT_LABEL'].drop('DEFECT_LABEL').sort_values(key=abs, ascending=False)
print(f"\nTop Features Correlated with DEFECT_LABEL:")
for feat, corr in target_corr.head(5).items():
    print(f"  {feat}: r = {corr:.3f}")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['darkgreen' if x > 0 else 'darkred' for x in target_corr.values]
target_corr.plot(kind='barh', color=colors, ax=ax)
ax.set_title('Feature Correlation with DEFECT_LABEL', fontsize=14, fontweight='bold')
ax.set_xlabel('Correlation Coefficient')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.savefig('06_target_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# STEP 4: FEATURE SELECTION

print("STEP 4: FEATURE SELECTION")

X = df_clean.drop('DEFECT_LABEL', axis=1)
y = df_clean['DEFECT_LABEL']

f_selector = SelectKBest(score_func=f_classif, k='all')
f_selector.fit(X, y)
f_scores = pd.DataFrame({'Feature': X.columns, 'F_Score': f_selector.scores_, 'P_Value': f_selector.pvalues_})

mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
mi_selector.fit(X, y)
mi_scores = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_selector.scores_})

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_scores = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})

combined_scores = f_scores[['Feature', 'F_Score']].merge(mi_scores, on='Feature').merge(rf_scores, on='Feature')
scaler = MinMaxScaler()
combined_scores[['F_Score_Norm', 'MI_Score_Norm', 'Importance_Norm']] = scaler.fit_transform(
    combined_scores[['F_Score', 'MI_Score', 'Importance']])
combined_scores['Average_Score'] = combined_scores[['F_Score_Norm', 'MI_Score_Norm', 'Importance_Norm']].mean(axis=1)
combined_scores = combined_scores.sort_values('Average_Score', ascending=False)

print("\nCombined Feature Ranking:")
print(combined_scores[['Feature', 'Average_Score']])

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(combined_scores))
width = 0.25
ax.bar(x - width, combined_scores['F_Score_Norm'], width, label='ANOVA F-test', color='steelblue')
ax.bar(x, combined_scores['MI_Score_Norm'], width, label='Mutual Information', color='coral')
ax.bar(x + width, combined_scores['Importance_Norm'], width, label='Random Forest', color='mediumseagreen')
ax.set_xlabel('Features')
ax.set_ylabel('Normalized Score')
ax.set_title('Feature Selection: Comparison of Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(combined_scores['Feature'], rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig('07_feature_selection_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

k = 7
top_features = combined_scores.head(k)['Feature'].tolist()
print(f"\nTop {k} Selected Features: {top_features}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(combined_scores.head(k)['Feature'], combined_scores.head(k)['Average_Score'], color='teal')
ax.set_xlabel('Average Normalized Score')
ax.set_title(f'Top {k} Selected Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('08_top_features.png', dpi=150, bbox_inches='tight')
plt.show()
