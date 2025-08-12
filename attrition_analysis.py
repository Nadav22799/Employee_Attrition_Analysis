# ============================================================================
# INTEGRATED EMPLOYEE ATTRITION ANALYSIS 
# Research Questions: What factors influence attrition + Can we predict at-risk employees?
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, f_oneway, kruskal
from scipy.stats import pearsonr, spearmanr
from scipy.stats import binom, chi2, f, t
from scipy.optimize import minimize
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
import itertools
from scipy.stats import shapiro, levene
import os

warnings.filterwarnings('ignore')

# Set up plotting style with bigger fonts
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 18
})
sns.set_palette("husl")

# Create Figures directory if it doesn't exist
figures_dir = "Figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)
    print(f"Created directory: {figures_dir}")

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING 
# ============================================================================
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("="*80)
print("COMPREHENSIVE EMPLOYEE ATTRITION ANALYSIS")
print("Research Questions: What factors influence attrition + predictive modeling")
print("="*80)

# Basic dataset overview
print(f"Dataset Overview:")
print(f"Total employees: {len(df)}")
print(f"Features: {df.shape[1]}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Convert attrition to binary
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
print(f"Attrition rate: {df['Attrition'].mean():.3f}")

# ============================================================================
# 2. COMPREHENSIVE FEATURE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("2. COMPREHENSIVE FEATURE STATISTICS WITH ATTRITION CONNECTION")
print("="*80)

# 2.1 Numerical and Ordinal Features Visualization
print("\n2.1 NUMERICAL AND ORDINAL FEATURES ANALYSIS:")
print("-" * 50)

numerical_features = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
                     'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
                     'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
                     'YearsSinceLastPromotion', 'YearsWithCurrManager']

ordinal_features = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
                   'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction',
                   'StockOptionLevel', 'WorkLifeBalance']

# Filter features that exist in the dataset
available_numerical = [f for f in numerical_features if f in df.columns]
available_ordinal = [f for f in ordinal_features if f in df.columns]
all_numeric_features = available_numerical

# Create comprehensive visualization for numerical/ordinal features
n_features = len(all_numeric_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_cols, n_rows, figsize=(6*n_rows, 20))
fig.suptitle('Feature Distributions by Attrition Status', fontsize=18, fontweight='bold', y=0.98)

# Flatten axes for easier indexing
if n_rows > 1:
    axes = axes.flatten()
elif n_cols > 1:
    axes = [axes] if n_features == 1 else axes
else:
    axes = [axes]

for i, feature in enumerate(all_numeric_features):
    if i < len(axes):
        ax = axes[i]

        # Create violin plots for better distribution visualization
        df_plot = df.copy()
        df_plot['Attrition_Status'] = df_plot['Attrition'].map({0: 'Stayed', 1: 'Left'})

        sns.violinplot(data=df_plot, x='Attrition_Status', y=feature, ax=ax, palette=sns.color_palette()[:2])

        ax.set_title(f'{feature}', fontsize=22, fontweight='bold', pad=15)
        ax.set_xlabel('Attrition Status', fontsize=18)
        ax.set_ylabel(feature, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)

# Hide unused subplots
for i in range(len(all_numeric_features), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.savefig(os.path.join(figures_dir, 'feature_distributions.png'), dpi=600, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nSummary Statistics:")

for feature in all_numeric_features:
    stayed_stats = df[df['Attrition'] == 0][feature].describe()
    left_stats = df[df['Attrition'] == 1][feature].describe()
    stayed_values = df[df['Attrition'] == 0][feature]
    left_values = df[df['Attrition'] == 1][feature]

    # Test for normality using Shapiro-Wilk test
    _, stayed_p_normal = shapiro(stayed_values)
    _, left_p_normal = shapiro(left_values)

    # Use parametric test if both groups are normally distributed (p > 0.05)
    # Use non-parametric test if either group is not normally distributed
    if stayed_p_normal > 0.05 and left_p_normal > 0.05:
        # Both groups are normally distributed - use t-test
        _, p_value = ttest_ind(stayed_values, left_values)
        test_used = "t-test (parametric)"
    else:
        # At least one group is not normally distributed - use Mann-Whitney U test
        _, p_value = mannwhitneyu(stayed_values, left_values, alternative='two-sided')
        test_used = "Mann-Whitney U (non-parametric)"

    print(f"Feature: {feature}")
    print(f"Mean - Stayed: {stayed_values.mean():.4f}, Left: {left_values.mean():.4f}")
    print(f"Normality - Stayed: p={stayed_p_normal:.4f}, Left: p={left_p_normal:.4f}")
    print(f"Test used: {test_used}")
    print(f"P-value: {p_value:.4f}")
    print("-" * 50)

# 2.2 Categorical Features Visualization
print("\n\n2.2 CATEGORICAL FEATURES ANALYSIS:")
print("-" * 50)

categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                       'JobRole', 'MaritalStatus', 'OverTime']

available_categorical = [f for f in categorical_features if f in df.columns] + available_ordinal

# Create visualization for categorical features
n_cat_features = len(available_categorical)
n_cols = 3
n_rows = (n_cat_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_cols, n_rows, figsize=(6*n_rows, 23))
fig.suptitle('Categorical Features Analysis by Attrition Status', fontsize=18, fontweight='bold', y=0.98)

# Flatten axes for easier indexing
if n_rows > 1:
    axes = axes.flatten()
elif n_cols > 1:
    axes = [axes] if n_cat_features == 1 else axes
else:
    axes = [axes]

for i, feature in enumerate(available_categorical):
    if i < len(axes):
        ax = axes[i]

        # Create grouped bar chart
        crosstab_pct = pd.crosstab(df[feature], df['Attrition'], normalize='index') * 100
        crosstab_pct.plot(kind='bar', ax=ax, color=[sns.color_palette()[1], sns.color_palette()[0]])

        ax.set_title(f'{feature}', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel(feature, fontsize=16)
        ax.set_ylabel('Percentage', fontsize=16)
        ax.legend(['Stayed', 'Left'], fontsize=16)
        ax.tick_params(axis='x', rotation=60, labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

# Hide unused subplots
for i in range(len(available_categorical), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(hspace=1, wspace=0.3)
plt.savefig(os.path.join(figures_dir, 'categorical_analysis.png'), dpi=600, bbox_inches='tight')
plt.show()

# Print categorical summary
print("\nCategorical Features Summary:")
for feature in available_categorical:
    crosstab = pd.crosstab(df[feature], df['Attrition'], margins=True)
    crosstab_pct = pd.crosstab(df[feature], df['Attrition'], normalize='index') * 100

    contingency = pd.crosstab(df[feature], df['Attrition'])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
        cramers_v = np.sqrt(chi2_stat / (len(df) * (min(contingency.shape) - 1)))
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    else:
        p_value, cramers_v, significance = 1.0, 0.0, ""

    print(f"\n{feature.upper()}: p={p_value:.4f}, Cramér's V={cramers_v:.3f}{significance}")
    for category in crosstab_pct.index[:-1]:  # Exclude 'All' row
        count = crosstab.loc[category, 1]  # Count of attrition=1
        total = crosstab.loc[category, 'All']  # Total count
        rate = crosstab_pct.loc[category, 1]  # Attrition rate
        print(f"  {category}: {rate:.1f}% attrition ({count}/{total})")

print("\n* p<0.05, ** p<0.01, *** p<0.001")

# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("3. CORRELATION ANALYSIS FOR ATTRITION PREDICTORS")
print("="*60)
print("Purpose: Identify strongest relationships with attrition to guide hypothesis testing")

# Encode categorical features for correlation analysis
df_encoded = df.copy()

# Binary categorical variables
df_encoded['OverTime_encoded'] = (df_encoded['OverTime'] == 'Yes').astype(int)
df_encoded['Gender_encoded'] = (df_encoded['Gender'] == 'Male').astype(int)

# Multi-category variables (using label encoding)
categorical_to_encode = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
for cat_feature in categorical_to_encode:
    if cat_feature in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[f'{cat_feature}_encoded'] = le.fit_transform(df_encoded[cat_feature])

# Create comprehensive list of predictors including encoded categorical features
encoded_categorical_features = ['OverTime_encoded', 'Gender_encoded'] + \
                              [f'{cat}_encoded' for cat in categorical_to_encode if cat in df.columns]

# Use numerical + ordinal + encoded categorical features + attrition for analysis
attrition_predictors = available_numerical + available_ordinal + encoded_categorical_features + ['Attrition']

# Filter features that actually exist in the dataset
attrition_predictors = [f for f in attrition_predictors if f in df_encoded.columns]

print(f"Total features in correlation analysis: {len(attrition_predictors)}")
print(f"- Numerical features: {len(available_numerical)}")
print(f"- Ordinal features: {len(available_ordinal)}")
print(f"- Encoded categorical features: {len([f for f in attrition_predictors if '_encoded' in f])}")

corr_matrix = df_encoded[attrition_predictors].corr()
print("\nCorrelations with Attrition (Top 15 strongest):")
print("-" * 60)
attrition_corrs = corr_matrix['Attrition'].drop('Attrition').sort_values(key=abs, ascending=False)

# Select top correlated features with attrition for readability (top 20 + Attrition)
top_features = attrition_corrs.head(20).index.tolist() + ['Attrition']
corr_matrix_subset = corr_matrix.loc[top_features, top_features]

# Create readable labels for the heatmap
readable_labels = []
for feature in top_features:
    if '_encoded' in feature:
        # Shorten categorical labels for readability
        clean_name = feature.replace('_encoded', '')
        readable_labels.append(clean_name)
    else:
        # Shorten long numerical feature names
        if 'YearsSinceLastPromotion' in feature:
            readable_labels.append('YearsSincePromo')
        elif 'YearsWithCurrManager' in feature:
            readable_labels.append('YearsWithManager')
        elif 'EnvironmentSatisfaction' in feature:
            readable_labels.append('EnvSatisfaction')
        elif 'RelationshipSatisfaction' in feature:
            readable_labels.append('RelSatisfaction')
        elif 'TrainingTimesLastYear' in feature:
            readable_labels.append('TrainingTimes')
        elif 'NumCompaniesWorked' in feature:
            readable_labels.append('NumCompanies')
        elif 'PercentSalaryHike' in feature:
            readable_labels.append('SalaryHike%')
        elif 'DistanceFromHome' in feature:
            readable_labels.append('Distance')
        elif 'MonthlyIncome' in feature:
            readable_labels.append('Income')
        elif 'TotalWorkingYears' in feature:
            readable_labels.append('TotalYears')
        else:
            readable_labels.append(feature)

plt.figure(figsize=(16, 14))
# Create custom colormap: white at 0, blue for negative, red for positive
colors = ['#0066CC', '#FFFFFF', '#CC0000']  # Blue, White, Red
n_bins = 100
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Create the heatmap WITHOUT numbers for better readability
mask = np.triu(np.ones_like(corr_matrix_subset, dtype=bool), k=1)  # Mask upper triangle
sns.heatmap(corr_matrix_subset,
            annot=False,  # Remove numbers for better readability
            cmap=cmap,
            center=0,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
            mask=mask,
            xticklabels=readable_labels,
            yticklabels=readable_labels,
            linewidths=0.5)

plt.title('Enhanced Correlation Matrix - Top 20 Attrition Predictors\n(Including All Feature Types: Blue=Negative, White=Zero, Red=Positive)',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('')
plt.ylabel('')
plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'correlation_matrix_enhanced.png'), dpi=600, bbox_inches='tight')
plt.show()

# Show top 15 correlations for readability
for var, corr in attrition_corrs.head(15).items():
    r, p = pearsonr(df_encoded[var], df_encoded['Attrition'])
    significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    # Clean display name
    display_name = var.replace('_encoded', '') if '_encoded' in var else var
    print(f"{display_name:25}: r = {corr:7.3f}, p = {p:.4f} {significance}")

print("\n* p<0.05, ** p<0.01, *** p<0.001")

# Create a focused correlation plot showing correlations with Attrition
plt.figure(figsize=(14, 10))
top_attrition_corrs = attrition_corrs.head(20)
readable_names = [name.replace('_encoded', '') if '_encoded' in name else
                 ('YearsSincePromo' if 'YearsSinceLastPromotion' in name else
                  'YearsWithManager' if 'YearsWithCurrManager' in name else
                  'EnvSatisfaction' if 'EnvironmentSatisfaction' in name else
                  'RelSatisfaction' if 'RelationshipSatisfaction' in name else
                  'TrainingTimes' if 'TrainingTimesLastYear' in name else
                  'NumCompanies' if 'NumCompaniesWorked' in name else
                  'SalaryHike%' if 'PercentSalaryHike' in name else
                  'Distance' if 'DistanceFromHome' in name else
                  'Income' if 'MonthlyIncome' in name else
                  'TotalYears' if 'TotalWorkingYears' in name else name)
                 for name in top_attrition_corrs.index]

colors = ['red' if x > 0 else 'blue' for x in top_attrition_corrs.values]
bars = plt.barh(range(len(top_attrition_corrs)), top_attrition_corrs.values,
               color=colors, alpha=0.7)
plt.yticks(range(len(top_attrition_corrs)), readable_names)
plt.xlabel('Correlation with Attrition', fontsize=12)
plt.title('Top 20 Features Correlation with Attrition\n(Red=Positive, Blue=Negative)\n→ These guide our hypothesis testing priorities',
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
            f'{width:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'top_attrition_correlations.png'), dpi=600, bbox_inches='tight')
plt.show()

print(f"\nKey Findings to Guide Hypothesis Testing:")
print(f"- Strongest positive correlation: {attrition_corrs.head(1).index[0].replace('_encoded', '')} (r = {attrition_corrs.head(1).values[0]:.3f})")
print(f"- Strongest negative correlation: {attrition_corrs.tail(1).index[0].replace('_encoded', '')} (r = {attrition_corrs.tail(1).values[0]:.3f})")
print(f"- We will prioritize testing these relationships in formal hypothesis tests")


# ============================================================================
# 4. EDUCATION AND DEPARTMENT HYPOTHESIS TESTING
# ============================================================================

def test_education_department_hypotheses(df):
    """
    RESEARCH QUESTION: Do education and department create non-obvious attrition patterns?
    FOCUS: Simple but interesting hypothesis tests for education and department effects
    BUSINESS VALUE: Actionable insights for targeted retention strategies
    """
    print("\n4. EDUCATION AND DEPARTMENT HYPOTHESIS TESTING")
    print("="*60)
    print("Research Focus: Testing education and department effects on attrition")
    
    results = {}
    
    # HYPOTHESIS 1: Education Paradox - Do higher educated employees leave more?
    # Non-trivial: Counter-intuitive that education might increase attrition
    print("\nHYPOTHESIS 1: EDUCATION PARADOX")
    print("-" * 40)
    print("H0: Higher education levels always lead to lower attrition")
    print("H1: Higher educated employees have higher attrition (seeking better opportunities)")
    
    # Calculate attrition rates by education level
    education_attrition = df.groupby('Education')['Attrition'].agg(['mean', 'count', 'std'])
    
    print(f"\nAttrition rates by education level:")
    education_labels = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
    for edu_level in sorted(education_attrition.index):
        rate = education_attrition.loc[edu_level, 'mean']
        n = education_attrition.loc[edu_level, 'count']
        label = education_labels.get(edu_level, f'Level {edu_level}')
        print(f"{label}: {rate:.3f} attrition rate (n={n})")
    
    # Test for differences using Kruskal-Wallis (appropriate for ordinal education levels)
    education_groups = [df[df['Education'] == level]['Attrition'] 
                       for level in sorted(df['Education'].unique())]
    h_stat, p_value = kruskal(*education_groups)
    
    # Test for paradox: Higher education (4,5) vs Lower education (1,2,3)
    higher_edu = df[df['Education'] >= 4]['Attrition']
    lower_edu = df[df['Education'] <= 3]['Attrition']
    
    if len(higher_edu) >= 10 and len(lower_edu) >= 10:
        _, p_paradox = mannwhitneyu(higher_edu, lower_edu, alternative='greater')
        
        higher_rate = higher_edu.mean()
        lower_rate = lower_edu.mean()
        
        paradox_detected = higher_rate > lower_rate and p_paradox < 0.05
        
        print(f"\nStatistical Results:")
        print(f"Kruskal-Wallis H-statistic: {h_stat:.3f}, p-value: {p_value:.4f}")
        print(f"Higher education rate: {higher_rate:.3f}")
        print(f"Lower education rate: {lower_rate:.3f}")
        print(f"Mann-Whitney U p-value (higher > lower): {p_paradox:.4f}")
        
        if paradox_detected:
            print("→ EDUCATION PARADOX CONFIRMED: Higher educated employees leave more!")
            print("→ Business insight: Highly educated employees may seek advancement opportunities elsewhere")
        else:
            print("→ No paradox: Education doesn't increase attrition risk")
        
        results['education_paradox'] = {
            'h_statistic': h_stat,
            'p_value_kruskal': p_value,
            'p_value_mannwhitney': p_paradox,
            'paradox_detected': paradox_detected,
            'higher_rate': higher_rate,
            'lower_rate': lower_rate
        }
    
    # HYPOTHESIS 2: Department Risk Inequality - Are some departments "attrition hotspots"?
    # Non-trivial: Testing if certain departments have dramatically different risk profiles
    print("\n\nHYPOTHESIS 2: DEPARTMENT RISK INEQUALITY")
    print("-" * 40)
    print("H0: All departments have similar attrition rates")
    print("H1: Some departments are 'attrition hotspots' with significantly higher rates")
    
    # Calculate attrition rates by department
    dept_attrition = df.groupby('Department')['Attrition'].agg(['mean', 'count', 'std'])
    
    print(f"\nAttrition rates by department:")
    for dept in dept_attrition.index:
        rate = dept_attrition.loc[dept, 'mean']
        n = dept_attrition.loc[dept, 'count']
        print(f"{dept}: {rate:.3f} attrition rate (n={n})")
    
    # Test for differences using Chi-Square Test of Independence (appropriate for categorical departments)
    contingency_table = pd.crosstab(df['Department'], df['Attrition'])
    chi2_stat, p_dept, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Cramer's V for effect size
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
    
    # Calculate coefficient of variation to measure inequality
    dept_rates = dept_attrition['mean']
    rate_std = dept_rates.std()
    rate_mean = dept_rates.mean()
    coefficient_of_variation = rate_std / rate_mean if rate_mean > 0 else 0
    
    # Identify hotspot: department with rate > 1.5 * overall average
    overall_rate = df['Attrition'].mean()
    hotspot_threshold = 1.5 * overall_rate
    
    hotspots = dept_rates[dept_rates > hotspot_threshold].index.tolist()
    safe_zones = dept_rates[dept_rates < 0.5 * overall_rate].index.tolist()
    
    inequality_detected = len(hotspots) > 0 and p_dept < 0.05
    
    print(f"\nStatistical Results:")
    print(f"Chi-Square statistic: {chi2_stat:.3f}, p-value: {p_dept:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"Cramer's V (effect size): {cramers_v:.3f}")
    print(f"Coefficient of variation: {coefficient_of_variation:.3f}")
    print(f"Overall attrition rate: {overall_rate:.3f}")
    print(f"Hotspot threshold (1.5x average): {hotspot_threshold:.3f}")
    
    if inequality_detected:
        print("→ DEPARTMENT INEQUALITY CONFIRMED: Significant differences exist!")
        if hotspots:
            print(f"→ ATTRITION HOTSPOTS: {', '.join(hotspots)}")
        if safe_zones:
            print(f"→ SAFE ZONES: {', '.join(safe_zones)}")
        print("→ Business insight: Focus retention efforts on hotspot departments")
    else:
        print("→ No significant department differences found")
    
    results['department_inequality'] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_dept,
        'degrees_of_freedom': dof,
        'cramers_v': cramers_v,
        'inequality_detected': inequality_detected,
        'hotspots': hotspots,
        'safe_zones': safe_zones,
        'coefficient_of_variation': coefficient_of_variation
    }
    
    # HYPOTHESIS 3: Job Role Risk Stratification - Do certain job roles have consistently higher attrition?
    # Non-trivial: Testing if specific roles like Sales Representatives are systematically high-risk
    print("\n\nHYPOTHESIS 3: JOB ROLE RISK STRATIFICATION")
    print("-" * 40)
    print("H0: All job roles have similar attrition rates")
    print("H1: Certain job roles (e.g., Sales Representatives) have significantly higher attrition")
    
    # Calculate attrition rates by job role
    role_attrition = df.groupby('JobRole')['Attrition'].agg(['mean', 'count', 'std'])
    
    print(f"\nAttrition rates by job role (sorted by rate):")
    role_attrition_sorted = role_attrition.sort_values('mean', ascending=False)
    for role in role_attrition_sorted.index:
        rate = role_attrition_sorted.loc[role, 'mean']
        n = role_attrition_sorted.loc[role, 'count']
        # print(f"{role}: {rate:.3f} attrition rate (n={n})")
    
    # Test for differences using Chi-Square Test of Independence (appropriate for categorical job roles)
    role_contingency_table = pd.crosstab(df['JobRole'], df['Attrition'])
    chi2_role, p_role, dof_role, expected_role = chi2_contingency(role_contingency_table)
    
    # Calculate Cramer's V for effect size
    n_role = role_contingency_table.sum().sum()
    cramers_v_role = np.sqrt(chi2_role / (n_role * (min(role_contingency_table.shape) - 1)))
    
    # Identify high-risk roles (>1.5x overall rate) and stable roles (<0.5x overall rate)
    role_rates = role_attrition['mean']
    overall_rate = df['Attrition'].mean()
    
    high_risk_roles = role_rates[role_rates > overall_rate * 1.5].index.tolist()
    stable_roles = role_rates[role_rates < overall_rate * 0.5].index.tolist()
    
    # Specific test for Sales Representative hypothesis
    sales_rep_rate = role_rates.get('Sales Representative', 0)
    sales_rep_hypothesis = sales_rep_rate > overall_rate * 1.2  # At least 20% higher than average
    
    # Test Sales Rep specifically against other roles using Fisher's Exact Test (more appropriate for specific comparison)
    if 'Sales Representative' in df['JobRole'].values:
        sales_rep_data = df[df['JobRole'] == 'Sales Representative']['Attrition']
        other_roles_data = df[df['JobRole'] != 'Sales Representative']['Attrition']
        
        if len(sales_rep_data) >= 5 and len(other_roles_data) >= 5:
            # Create 2x2 contingency table for Fisher's exact test
            sales_left = sales_rep_data.sum()
            sales_stayed = len(sales_rep_data) - sales_left
            others_left = other_roles_data.sum()
            others_stayed = len(other_roles_data) - others_left
            
            contingency_2x2 = [[sales_left, sales_stayed], [others_left, others_stayed]]
            
            from scipy.stats import fisher_exact
            odds_ratio, p_fisher = fisher_exact(contingency_2x2)
            
            sales_rep_significant = p_fisher < 0.05 and odds_ratio > 1.2
        else:
            odds_ratio, p_fisher, sales_rep_significant = 0, 1, False
    else:
        sales_rep_rate, odds_ratio, p_fisher, sales_rep_significant = 0, 0, 1, False
    
    role_stratification_detected = p_role < 0.05 and len(high_risk_roles) > 0
    
    print(f"\nStatistical Results:")
    print(f"Chi-Square statistic: {chi2_role:.3f}, p-value: {p_role:.4f}")
    print(f"Degrees of freedom: {dof_role}")
    print(f"Cramer's V (effect size): {cramers_v_role:.3f}")
    print(f"Overall attrition rate: {overall_rate:.3f}")
    
    if 'Sales Representative' in role_rates.index:
        print(f"Sales Representative rate: {sales_rep_rate:.3f}")
        print(f"Fisher's exact test p-value: {p_fisher:.4f}")
        print(f"Odds ratio (Sales Rep vs Others): {odds_ratio:.2f}")
    
    if role_stratification_detected:
        print("→ JOB ROLE STRATIFICATION CONFIRMED: Significant differences exist!")
        if high_risk_roles:
            print(f"→ HIGH-RISK ROLES: {', '.join(high_risk_roles[:3])}")  # Show top 3
        if stable_roles:
            print(f"→ STABLE ROLES: {', '.join(stable_roles[:3])}")  # Show top 3
        
        if sales_rep_significant:
            print(f"→ SALES REPRESENTATIVE HYPOTHESIS CONFIRMED: {odds_ratio:.1f}x higher odds of attrition!")
        
        print("→ Business insight: Role-specific retention programs needed")
    else:
        print("→ No significant role stratification found")
    
    results['job_role_stratification'] = {
        'chi2_statistic': chi2_role,
        'p_value': p_role,
        'degrees_of_freedom': dof_role,
        'cramers_v': cramers_v_role,
        'stratification_detected': role_stratification_detected,
        'high_risk_roles': high_risk_roles,
        'stable_roles': stable_roles,
        'sales_rep_rate': sales_rep_rate,
        'sales_rep_odds_ratio': odds_ratio,
        'sales_rep_p_value': p_fisher,
        'sales_rep_hypothesis_confirmed': sales_rep_significant
    }
    
    return results

# ============================================================================
# VISUALIZATION FOR EDUCATION AND DEPARTMENT PATTERNS
# ============================================================================

def create_job_role_visualization(df, results=None, save_path=None):
    """Create job role risk stratification visualization"""
    
    print(f"\nCREATING JOB ROLE RISK STRATIFICATION VISUALIZATION...")
    
    # Create single subplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Job Role Risk Stratification Analysis', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Job Role Risk Stratification
    role_rates = df.groupby('JobRole')['Attrition'].mean().sort_values(ascending=False)
    overall_rate = df['Attrition'].mean()
    
    # Color code roles based on risk level
    colors = []
    for rate in role_rates.values:
        if rate > overall_rate * 1.5:
            colors.append('red')  # High risk
        elif rate < overall_rate * 0.5:
            colors.append('green')  # Safe
        else:
            colors.append('steelblue')  # Normal
    
    # Create horizontal bar chart for better role name readability
    bars = ax.barh(range(len(role_rates)), role_rates.values,
                   color=colors, alpha=0.8, edgecolor='black')
    
    # Highlight Sales Representative if present
    if 'Sales Representative' in role_rates.index:
        sales_idx = list(role_rates.index).index('Sales Representative')
        bars[sales_idx].set_edgecolor('black')
        bars[sales_idx].set_linewidth(3)
    
    # Add reference lines
    ax.axvline(x=overall_rate, color='black', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Overall Rate ({overall_rate:.3f})')
    ax.axvline(x=overall_rate * 1.5, color='red', linestyle=':', alpha=0.7, linewidth=2,
               label='High Risk Threshold')
    ax.axvline(x=overall_rate * 0.5, color='green', linestyle=':', alpha=0.7, linewidth=2,
               label='Safe Zone Threshold')
    
    ax.set_title('Job Role Attrition Risk Assessment\nH1: Sales Representatives = High Risk', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Attrition Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Job Role', fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(role_rates)))
    ax.set_yticklabels([role[:30] + '...' if len(role) > 30 else role 
                        for role in role_rates.index], fontsize=10)
    ax.tick_params(axis='x', labelsize=11)
    ax.legend(fontsize=10, loc='lower right')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create Figures directory if it doesn't exist
    figures_dir = 'Figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save figure
    if save_path:
        # If save_path doesn't include directory, save to Figures folder
        if not os.path.dirname(save_path):
            save_path = os.path.join(figures_dir, save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Figure saved to: {save_path}")
    else:
        default_path = os.path.join(figures_dir, 'job_role_risk_stratification.png')
        plt.savefig(default_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Figure saved as: {default_path}")
    
    plt.show()

# ============================================================================
# EXECUTE EDUCATION, DEPARTMENT, AND JOB ROLE HYPOTHESIS TESTING
# ============================================================================

print("\nEXECUTING EDUCATION, DEPARTMENT, AND JOB ROLE HYPOTHESIS TESTING...")
hypothesis_results = test_education_department_hypotheses(df)
create_job_role_visualization(df, save_path='job_role_analysis.png')

print("\n" + "="*60)
print("EDUCATION, DEPARTMENT, AND JOB ROLE HYPOTHESIS TESTING COMPLETE")
print("="*60)

# ============================================================================
# 5. ENHANCED MULTIPLE COMPARISONS CORRECTION - COMPREHENSIVE VERSION
# ============================================================================

print("\n" + "="*80)
print("ENHANCED MULTIPLE COMPARISONS CORRECTION - COMPREHENSIVE")
print("="*80)
print("Testing ALL features (numerical + ordinal + categorical) with appropriate statistical tests")

# Use the original feature lists from your code
all_features_for_testing = available_numerical + available_ordinal + encoded_categorical_features
print(f"Total features to test: {len(all_features_for_testing)}")
print(f"- Numerical features: {len(available_numerical)}")
print(f"- Ordinal features: {len(available_ordinal)}")
print(f"- Categorical features: {len(encoded_categorical_features)}")

# Store all test results
all_test_results = []

# 1. TEST NUMERICAL AND ORDINAL FEATURES
print(f"\n1. TESTING NUMERICAL AND ORDINAL FEATURES:")
print("-" * 60)

for feature in available_numerical + available_ordinal:
    try:
        # Split data by attrition
        stayed_values = df[df['Attrition'] == 0][feature].dropna()
        left_values = df[df['Attrition'] == 1][feature].dropna()
        
        if len(stayed_values) < 5 or len(left_values) < 5:
            continue
        
        # Test for normality (using sample if data is large)
        sample_size = min(5000, len(stayed_values))
        _, stayed_p_normal = shapiro(stayed_values.sample(sample_size) if sample_size < len(stayed_values) else stayed_values)
        sample_size = min(5000, len(left_values))
        _, left_p_normal = shapiro(left_values.sample(sample_size) if sample_size < len(left_values) else left_values)
        
        # Choose appropriate test based on normality
        if stayed_p_normal > 0.05 and left_p_normal > 0.05:
            # Both groups are normal - use t-test
            _, p_value = ttest_ind(stayed_values, left_values)
            test_used = "t-test"
            
            # Calculate Cohen's d
            pooled_std = np.sqrt(((len(stayed_values)-1)*stayed_values.var() + 
                                 (len(left_values)-1)*left_values.var()) / 
                                (len(stayed_values) + len(left_values) - 2))
            effect_size = (left_values.mean() - stayed_values.mean()) / pooled_std
            effect_type = "Cohen's d"
        else:
            # Use non-parametric Mann-Whitney U
            _, p_value = mannwhitneyu(stayed_values, left_values, alternative='two-sided')
            test_used = "Mann-Whitney U"
            
            # Calculate rank-biserial correlation as effect size
            u_statistic = mannwhitneyu(stayed_values, left_values, alternative='two-sided')[0]
            effect_size = 1 - (2 * u_statistic) / (len(stayed_values) * len(left_values))
            effect_type = "Rank-biserial r"
        
        # Also calculate correlation
        corr_coef, corr_p = pearsonr(df[feature], df['Attrition'])
        
        # Store results
        all_test_results.append({
            'Feature': feature,
            'Type': 'Numerical' if feature in available_numerical else 'Ordinal',
            'Test': test_used,
            'P_Value': p_value,
            'Effect_Size': effect_size,
            'Effect_Type': effect_type,
            'Correlation': corr_coef,
            'Correlation_P': corr_p,
            'Stayed_Mean': stayed_values.mean(),
            'Left_Mean': left_values.mean(),
            'Stayed_N': len(stayed_values),
            'Left_N': len(left_values)
        })
        
        print(f"{feature:25}: {test_used:15} p={p_value:.6f}, {effect_type}={effect_size:.3f}")
        
    except Exception as e:
        print(f"Error testing {feature}: {str(e)}")
        continue

# 2. TEST CATEGORICAL FEATURES
print(f"\n2. TESTING CATEGORICAL FEATURES:")
print("-" * 60)

for feature in encoded_categorical_features:
    try:
        # Create contingency table
        contingency = pd.crosstab(df_encoded[feature], df['Attrition'])
        
        # Check if chi-square assumptions are met (expected frequencies >= 5)
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
        
        # Skip if expected frequencies are too low
        if (expected < 5).sum() > expected.size * 0.2:  # More than 20% of cells < 5
            print(f"{feature:25}: Skipped (expected frequencies < 5)")
            continue
        
        # Calculate Cramér's V as effect size
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))
        
        # Store results
        all_test_results.append({
            'Feature': feature,
            'Type': 'Categorical',
            'Test': 'Chi-square',
            'P_Value': p_value,
            'Effect_Size': cramers_v,
            'Effect_Type': "Cramér's V",
            'Correlation': np.nan,
            'Correlation_P': np.nan,
            'Stayed_Mean': np.nan,
            'Left_Mean': np.nan,
            'Stayed_N': contingency.iloc[:, 0].sum(),
            'Left_N': contingency.iloc[:, 1].sum()
        })
        
        print(f"{feature:25}: Chi-square      p={p_value:.6f}, Cramér's V={cramers_v:.3f}")
        
    except Exception as e:
        print(f"Error testing {feature}: {str(e)}")
        continue

# Convert to DataFrame for easier manipulation
results_df = pd.DataFrame(all_test_results)

if len(results_df) == 0:
    print("No valid tests were performed!")
else:
    print(f"\n3. APPLYING MULTIPLE COMPARISONS CORRECTIONS:")
    print("-" * 60)
    
    # Get p-values for correction
    p_values = results_df['P_Value'].values
    n_tests = len(p_values)
    
    print(f"Total tests performed: {n_tests}")
    print(f"Original α = 0.05")
    print(f"Bonferroni-corrected α = {0.05/n_tests:.8f}")
    
    # Apply various correction methods
    # 1. Bonferroni
    bonferroni_alpha = 0.05 / n_tests
    results_df['Significant_Bonferroni'] = p_values < bonferroni_alpha
    
    # 2. False Discovery Rate (Benjamini-Hochberg) - RECOMMENDED
    fdr_rejected, fdr_corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['FDR_P_Value'] = fdr_corrected_p
    results_df['Significant_FDR'] = fdr_rejected
    
    # 3. Holm-Bonferroni (step-down)
    holm_rejected, holm_corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='holm')
    results_df['Holm_P_Value'] = holm_corrected_p
    results_df['Significant_Holm'] = holm_rejected
    
    # 4. Šidák correction
    sidak_rejected, sidak_corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='sidak')
    results_df['Sidak_P_Value'] = sidak_corrected_p
    results_df['Significant_Sidak'] = sidak_rejected
    
    # Sort by original p-value
    results_df = results_df.sort_values('P_Value').reset_index(drop=True)
    
    # Summary statistics
    uncorrected_sig = sum(p_values < 0.05)
    bonferroni_sig = sum(results_df['Significant_Bonferroni'])
    fdr_sig = sum(results_df['Significant_FDR'])
    holm_sig = sum(results_df['Significant_Holm'])
    sidak_sig = sum(results_df['Significant_Sidak'])
    
    print(f"\nCORRECTION RESULTS SUMMARY:")
    print(f"Uncorrected (α=0.05):     {uncorrected_sig:2d} significant")
    print(f"Bonferroni:               {bonferroni_sig:2d} significant")
    print(f"FDR (Benjamini-Hochberg): {fdr_sig:2d} significant ← RECOMMENDED")
    print(f"Holm-Bonferroni:          {holm_sig:2d} significant")
    print(f"Šidák:                    {sidak_sig:2d} significant")
    
    # 4. DETAILED RESULTS TABLE
    print(f"\n4. TOP SIGNIFICANT RESULTS (FDR-corrected):")
    print("-" * 100)
    
    # Show FDR-significant results
    fdr_significant = results_df[results_df['Significant_FDR']].copy()
    
    if len(fdr_significant) > 0:
        print("ROBUST FINDINGS (FDR-corrected, recommended for business decisions):")
        print("Feature                   Type          Test            P-Value    FDR P-Value  Effect Size")
        print("-" * 100)
        
        for _, row in fdr_significant.head(15).iterrows():
            print(f"{row['Feature']:25} {row['Type']:12} {row['Test']:15} "
                  f"{row['P_Value']:10.6f} {row['FDR_P_Value']:10.6f} "
                  f"{row['Effect_Size']:8.3f} ({row['Effect_Type']})")
    else:
        print("No features remain significant after FDR correction!")
    
    print(f"\n5. ALL RESULTS TABLE (Top 20 by significance):")
    print("-" * 120)
    print("Feature                   Type          P-Value    FDR_Sig  Bonf_Sig  Effect_Size  Stayed_Mean  Left_Mean")
    print("-" * 120)
    
    for _, row in results_df.head(20).iterrows():
        fdr_sig = "Yes" if row['Significant_FDR'] else "No"
        bonf_sig = "Yes" if row['Significant_Bonferroni'] else "No"
        stayed_mean = f"{row['Stayed_Mean']:.2f}" if not pd.isna(row['Stayed_Mean']) else "N/A"
        left_mean = f"{row['Left_Mean']:.2f}" if not pd.isna(row['Left_Mean']) else "N/A"
        
        print(f"{row['Feature']:25} {row['Type']:12} {row['P_Value']:10.6f} "
              f"{fdr_sig:7} {bonf_sig:8} {row['Effect_Size']:10.3f} "
              f"{stayed_mean:10} {left_mean:10}")

# ENHANCED VISUALIZATION: Multiple Comparisons Correction Analysis
print(f"\nCREATING ENHANCED VISUALIZATIONS FOR MULTIPLE COMPARISONS ANALYSIS...")

if len(results_df) > 0:
    # Verify all counts are integers and debug
    significant_counts = [
        int(np.sum(p_values < 0.05)),
        int(np.sum(results_df['Significant_Bonferroni'].values)), 
        int(np.sum(results_df['Significant_FDR'].values)), 
        int(np.sum(results_df['Significant_Holm'].values)), 
        int(np.sum(results_df['Significant_Sidak'].values))
    ]
    
    # Debug print to check values
    print(f"Debug - Significant counts: {significant_counts}")
    print(f"Debug - Types: {[type(x) for x in significant_counts]}")
    
    # Create streamlined multiple comparisons visualization - TOP ROW ONLY
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Enhanced Multiple Comparisons Correction Analysis\nAll Features: Numerical + Ordinal + Categorical',
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. P-value Distribution (Raw vs Corrected)
    axes[0].hist(results_df['P_Value'], bins=20, alpha=0.7, color='lightblue',
                edgecolor='navy', label='Raw p-values')
    axes[0].hist(results_df['FDR_P_Value'], bins=20, alpha=0.7, color='lightcoral',
                edgecolor='darkred', label='FDR corrected')
    axes[0].axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    axes[0].axvline(x=bonferroni_alpha, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Bonferroni α = {bonferroni_alpha:.4f}')
    axes[0].set_title('P-value Distribution\n(Raw vs FDR Corrected)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('P-value', fontsize=14)
    axes[0].set_ylabel('Frequency', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Correction Method Comparison - Fixed version
    try:
        correction_methods = ['Uncorrected', 'Bonferroni', 'FDR (B-H)', 'Holm', 'Šidák']
        colors = ['lightgray', 'lightcoral', 'lightgreen', 'orange', 'lightblue']
        bars = axes[1].bar(correction_methods, significant_counts,
                           color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.3,
                        f'{int(height)}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        axes[1].set_title('Significant Results by Correction Method\n(Type I Error Control)', 
                         fontsize=16, fontweight='bold')
        axes[1].set_ylabel('Number of Significant Results', fontsize=14)
        axes[1].set_xlabel('Correction Method', fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        axes[1].text(0.5, 0.5, f'Bar chart error:\n{str(e)}', 
                    transform=axes[1].transAxes, ha='center', va='center')
        axes[1].set_title('Correction Method Comparison (Error)', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'enhanced_multiple_comparisons_analysis.png'), 
                dpi=600, bbox_inches='tight')
    plt.show()

# BUSINESS INSIGHTS SUMMARY
if len(results_df) > 0:
    print(f"\n" + "="*80)
    print("BUSINESS-READY INSIGHTS AFTER MULTIPLE COMPARISONS CORRECTION")
    print("="*80)
    
    fdr_significant_features = results_df[results_df['Significant_FDR'] == True]
    
    print(f"STATISTICALLY ROBUST FINDINGS:")
    print(f"• {len(fdr_significant_features)} features have genuine associations with attrition")
    print(f"• False Discovery Rate controlled at 5%")
    print(f"• Results are suitable for business decision-making")
    
    if len(fdr_significant_features) > 0:
        print(f"\nTOP ACTIONABLE FEATURES (FDR-corrected):")
        for i, (_, row) in enumerate(fdr_significant_features.head(10).iterrows(), 1):
            feature_name = row['Feature']
            effect = row['Effect_Size']
            
            if row['Type'] in ['Numerical', 'Ordinal']:
                direction = "Higher" if row['Left_Mean'] > row['Stayed_Mean'] else "Lower"
                print(f"{i:2d}. {feature_name}: {direction} values associated with attrition "
                      f"(effect size: {effect:.3f})")
            else:
                print(f"{i:2d}. {feature_name}: Significant association with attrition "
                      f"(Cramér's V: {effect:.3f})")
        
        print(f"\nRECOMMENDAITONS:")
        print(f"• Focus retention efforts on these {len(fdr_significant_features)} validated factors")
        print(f"• Prioritize factors with largest effect sizes")
        print(f"• These results control for multiple testing - no false positives expected")
    else:
        print(f"\nSURPRISING RESULT: No features remain significant after correction!")
        print(f"This suggests:")
        print(f"• Original findings may have been due to multiple testing")
        print(f"• Effect sizes are smaller than initially thought")
        print(f"• Consider less stringent FDR threshold or larger sample size")

print(f"\nMULTIPLE COMPARISONS CORRECTION COMPLETE!")
print(f"Use FDR-significant features for reliable business insights.")
# ============================================================================
# 6. PREDICTIVE MODELING
# ============================================================================

print("\n" + "="*60)
print("6. PREDICTIVE MODELING FOR EMPLOYEE ATTRITION")
print("="*60)

# Prepare features for modeling
print("6.1 Feature Engineering and Preparation:")

# Use encoded features from correlation analysis
X = df_encoded[encoded_categorical_features + available_ordinal + available_numerical].copy()
y = df['Attrition']

print(f"Features used: {len(X.columns)} features")
print(f"Numerical features: {len([col for col in X.columns if not col.endswith('_encoded')])}")
print(f"Categorical features: {len([col for col in X.columns if col.endswith('_encoded')])}")
print(f"Target variable: Attrition (0=Stayed, 1=Left)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6.2 Logistic Regression
print("\n6.2 Logistic Regression Model:")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

accuracy_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print(f"Accuracy: {accuracy_lr:.4f}")
print(f"AUC-ROC: {auc_lr:.4f}")

# Feature importance from coefficients
feature_importance_lr = pd.DataFrame({
    'feature': X.columns,
    'coefficient': log_reg.coef_[0],
    'abs_coefficient': np.abs(log_reg.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(f"\nTop 5 Most Important Features (Logistic Regression):")
for i, row in feature_importance_lr.head().iterrows():
    direction = "increases" if row['coefficient'] > 0 else "decreases"
    print(f"{row['feature']:25}: {row['coefficient']:7.3f} ({direction} attrition risk)")

# 6.3 Random Forest
print("\n6.3 Random Forest Model:")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"Accuracy: {accuracy_rf:.4f}")
print(f"AUC-ROC: {auc_rf:.4f}")

# Feature importance from Random Forest
rf_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Most Important Features (Random Forest):")
for i, row in rf_importance.head().iterrows():
    print(f"{row['feature']:25}: {row['importance']:7.3f}")


# ============================================================================
# 7. COMPREHENSIVE MODEL EVALUATION (ENHANCED from old.py)
# ============================================================================

print("\n" + "="*60)
print("7. MODEL EVALUATION AND COMPARISON")
print("="*60)

# Classification reports
print("7.1 Detailed Classification Results:")
print("\nLogistic Regression:")
print(classification_report(y_test, y_pred_lr, target_names=['Stayed', 'Left']))

print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf, target_names=['Stayed', 'Left']))

# ROC Curve comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

axes[0].plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f}, Accuracy = {accuracy_lr:.3f})', linewidth=3)
axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f}, Accuracy = {accuracy_rf:.3f})', linewidth=3)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier', linewidth=2)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Feature importance comparison (top 10)
top_features = rf_importance.head(10)
bars = axes[1].barh(range(len(top_features)), top_features['importance'])
axes[1].set_yticks(range(len(top_features)))
axes[1].set_yticklabels(top_features['feature'], fontsize=11)
axes[1].set_xlabel('Importance Score', fontsize=12)
axes[1].set_title('Top 10 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    axes[1].text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'roc_comparison.png'), dpi=600, bbox_inches='tight')
plt.show()

# Risk stratification
print("\n7.2 Risk Stratification Analysis:")
y_pred_proba_best = y_pred_proba_rf  # Use Random Forest probabilities

# Create risk categories
risk_categories = pd.cut(y_pred_proba_best,
                        bins=[0, 0.1, 0.3, 0.7, 1.0],
                        labels=['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'])

risk_analysis = pd.DataFrame({
    'Risk_Category': risk_categories,
    'Actual_Attrition': y_test
})

risk_summary = risk_analysis.groupby('Risk_Category')['Actual_Attrition'].agg(['count', 'sum', 'mean']).round(3)
risk_summary.columns = ['Total_Employees', 'Actual_Departures', 'Attrition_Rate']

print("Risk Stratification Results:")
print(risk_summary)

# VISUALIZATION: Enhanced Risk Stratification
print(f"\nCREATING ENHANCED RISK STRATIFICATION VISUALIZATIONS...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Enhanced Risk Stratification Analysis - Predictive Model Performance',
             fontsize=22, fontweight='bold', y=0.98)

# 1. Risk Category Distribution with Actual Outcomes
risk_counts = risk_analysis['Risk_Category'].value_counts().sort_index()
actual_left = risk_analysis.groupby('Risk_Category')['Actual_Attrition'].sum().sort_index()
actual_stayed = risk_counts - actual_left

x_pos = np.arange(len(risk_counts))
width = 0.6

# Stacked bar chart
bars1 = axes[0].bar(x_pos, actual_stayed, width, label='Stayed',
                     color='lightblue', alpha=0.8)
bars2 = axes[0].bar(x_pos, actual_left, width, bottom=actual_stayed,
                     label='Left', color='lightcoral', alpha=0.8)

axes[0].set_title('Employee Distribution by Risk Category\n(Actual Outcomes)',
                   fontsize=18, fontweight='bold')
axes[0].set_xlabel('Risk Category', fontsize=20)
axes[0].set_ylabel('Number of Employees', fontsize=20)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(risk_counts.index, rotation=45, ha='right', fontsize=18)
axes[0].tick_params(axis='y', which='major', labelsize=18)
axes[0].legend(fontsize=14)

# Add total count labels
for i, (stayed, left) in enumerate(zip(actual_stayed, actual_left)):
    total = stayed + left
    axes[0].text(i, total + 5, f'{total}', ha='center', va='bottom',
                  fontsize=14, fontweight='bold')

# 2. Prediction Accuracy by Risk Category
if len(risk_summary) > 0:
    predicted_rates = []
    actual_rates = risk_summary['Attrition_Rate'].values
    categories = risk_summary.index

    # Calculate average predicted probability for each risk category
    for category in categories:
        category_mask = risk_analysis['Risk_Category'] == category
        if category_mask.sum() > 0:
            avg_pred_prob = y_pred_proba_best[category_mask].mean()
            predicted_rates.append(avg_pred_prob)
        else:
            predicted_rates.append(0)

    x_accuracy = np.arange(len(categories))
    width = 0.35

    bars3 = axes[1].bar(x_accuracy - width/2, actual_rates, width,
                         label='Actual Rate', alpha=0.8, color='green')
    bars4 = axes[1].bar(x_accuracy + width/2, predicted_rates, width,
                         label='Predicted Rate', alpha=0.8, color='orange')

    axes[1].set_title('Prediction Calibration by Risk Category\n(Predicted vs Actual)',
                       fontsize=18, fontweight='bold')
    axes[1].set_xlabel('Risk Category', fontsize=20)
    axes[1].set_ylabel('Attrition Rate', fontsize=20)
    axes[1].set_xticks(x_accuracy)
    axes[1].set_xticklabels(categories, rotation=45, ha='right', fontsize=18)
    axes[1].tick_params(axis='y', which='major', labelsize=18)
    axes[1].legend(fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(figures_dir, 'enhanced_risk_stratification.png'), dpi=600, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS COMPLETE WITH RESEARCH-ALIGNED ENHANCEMENTS")
print("="*80)