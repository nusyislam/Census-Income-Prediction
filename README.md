# Census Income Prediction Model
## Overview
Machine learning solution predicting whether an individual's income exceeds $50K/year based on demographic and employment features from the 1994 U.S. Census dataset. Implements full ML lifecycle from data exploration to model deployment.

## Installation
Clone repository
```bash
git clone https://github.com/yourusername/census-income-prediction.git
```
Navigate to project directory
```bash
cd census-income-prediction
```
Install dependencies
```bash
pip install -r requirements.txt
```
## Dependencies:
- Python 3.8+
- pandas
- scikit-learn
- matplotlib
- seaborn

## Objectives
1. Predict income brackets with >70% F1-score
2. Identify key socioeconomic factors influencing income
3. Create reusable model for financial service applications
4. Demonstrate end-to-end ML workflow

## Dataset
1994 U.S. Census data containing:
- 32,561 instances
- 15 features (age, education, occupation, etc.)
- Target variable: income_binary (<=50K or >50K)

## Technical Approach:
1. Data Preprocessing:
   - Missing value imputation (median/mode)
   - One-hot encoding for categorical features
   - Standard scaling for numerical features
2. Modeling:
   - Primary: Random Forest
   - Baseline: Logistic Regression
   - Evaluation: F1-score, ROC AUC, Precision-Recall curves
3. Optimization:
   - Class weighting for imbalance handling
   - GridSearchCV for hyperparameter tuning

## Results
Model	          F1-Score	 ROC AUC	  Precision
Random Forest	  0.72	     0.89	    0.59
Logistic Reg	  0.65	     0.88	    0.61
Key Insights:

- Top 5 predictors:
  1. Education level
  2. Age
  3. Weekly work hours
  4. Occupation type
  5. Marital status
