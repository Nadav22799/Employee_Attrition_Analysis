# Employee Attrition Analysis Project

## 🎯 Project Overview

This project provides a comprehensive statistical analysis of employee attrition using the IBM HR Analytics Employee Attrition dataset. The analysis combines exploratory data analysis, advanced statistical testing, and predictive modeling to answer two key research questions:

1. **What factors significantly influence employee attrition?**
2. **Can we build predictive models to identify at-risk employees?**

## 📁 Project Structure

```
├── README.md                              # This file
├── requirements.txt                       # Python package dependencies
├── WA_Fn-UseC_-HR-Employee-Attrition.csv # Dataset
├── attrition_analysis.py                 # ⭐ MAIN FILE - Integrated comprehensive analysis
└── Figures/                              # Output directory (created automatically)
    ├── feature_distributions.png         # Feature analysis by attrition status
    ├── categorical_analysis.png          # Categorical feature breakdown
    ├── correlation_matrix_enhanced.png   # Feature correlation heatmap
    ├── top_attrition_correlations.png    # Top correlations with attrition
    ├── job_role_analysis.png             # Job role risk stratification
    ├── multiple_comparisons_analysis.png # Statistical rigor analysis
    ├── roc_comparison.png                # Model performance comparison
    └── enhanced_risk_stratification.png  # Business impact analysis
```

## 🔬 Analysis Features

### **attrition_analysis.py** - The Complete Analysis Pipeline

#### **1. Data Loading & Preprocessing**
- Dataset overview and quality assessment
- Attrition rate calculation and data cleaning
- Feature type classification and encoding

#### **2. Comprehensive Feature Analysis**
- **Numerical Features**: Violin plots showing distributions by attrition status
- **Categorical/Ordinal Features**: Grouped bar charts with percentage breakdowns
- **Statistical Testing**: Normality tests, Mann-Whitney U, Chi-square tests
- **Effect Size Calculations**: Cohen's d, Cramér's V for practical significance

#### **3. Correlation Analysis**
- Enhanced correlation matrix with top 20 attrition predictors
- Color-coded heatmap (blue=negative, red=positive correlations)
- Top correlations visualization to guide hypothesis testing priorities

#### **4. Education & Department Hypothesis Testing**
- **Education Paradox**: Tests if higher education leads to higher attrition
- **Department Risk Inequality**: Identifies "attrition hotspots" across departments
- **Job Role Risk Stratification**: Analyzes role-specific attrition patterns
- **Fisher's Exact Tests**: For specific hypotheses (e.g., Sales Representatives)

#### **5. Multiple Comparisons Correction**
- **Comprehensive Testing**: All numerical, ordinal, and categorical features
- **Four Correction Methods**: Bonferroni, FDR (Benjamini-Hochberg), Holm, Šidák
- **Effect Size Preservation**: Maintains practical significance alongside statistical significance
- **Business-Ready Results**: FDR-corrected findings for reliable decision-making

#### **6. Predictive Modeling**
- **Logistic Regression**: With feature scaling and coefficient interpretation
- **Random Forest**: With feature importance rankings
- **Model Comparison**: ROC curves, accuracy, AUC metrics
- **Risk Stratification**: 4-tier risk categories (Low, Medium, High, Very High)

## 🛠️ Requirements & Installation

### **System Requirements**
- Python 3.7+ 
- Operating System: Windows, macOS, or Linux

### **Required Python Packages**

```bash
# Core data science packages
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0

# Machine learning
scikit-learn>=1.0.0

# Statistical analysis
statsmodels>=0.12.0
```

### **Installation Methods**

#### **Option 1: Using pip (Recommended)**
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

#### **Option 2: Using conda**
```bash
conda install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

```bash
pip install -r requirements.txt
```

#### **Option 3: Ubuntu/Debian users**
```bash
sudo apt update
sudo apt install python3-pandas python3-numpy python3-matplotlib python3-seaborn python3-scipy python3-sklearn python3-statsmodels
```

## 🚀 How to Run

### **Step 1: Verify Dataset**
Ensure `WA_Fn-UseC_-HR-Employee-Attrition.csv` is in the same directory as `attrition_analysis.py`.

### **Step 2: Run the Analysis**
```bash
python3 attrition_analysis.py
```

### **Step 3: Check Outputs**
The analysis will create a `Figures/` directory with 8 visualization files:

1. `Figures/feature_distributions.png` - Feature analysis by attrition status
2. `Figures/categorical_analysis.png` - Categorical feature breakdown  
3. `Figures/correlation_matrix_enhanced.png` - Feature correlation heatmap
4. `Figures/top_attrition_correlations.png` - Top correlations with attrition
5. `Figures/job_role_analysis.png` - Job role risk stratification analysis
6. `Figures/multiple_comparisons_analysis.png` - Statistical rigor analysis
7. `Figures/roc_comparison.png` - Model performance comparison
8. `Figures/enhanced_risk_stratification.png` - Business impact analysis

---

