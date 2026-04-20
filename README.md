# Loan Default Prediction

## Project Overview
This project builds and compares 5 machine learning models to predict whether a loan applicant will default. The dataset contains over 300,000 loan applications with 122 features.

## Models Implemented
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

## Key Techniques
- **SMOTE** for handling severe class imbalance (91.9% repaid vs 8.1% default)
- **SHAP** for model explainability
- **Differential Privacy** to analyze privacy-accuracy trade-offs
- **Bootstrap confidence intervals** for statistical significance

## Results Summary

| Model | ROC-AUC | F1-Score | Recall |
|-------|---------|----------|--------|
| Logistic Regression | 0.7421 | 68.32% | 65.41% |
| Decision Tree | 0.7156 | 66.89% | 63.22% |
| Random Forest | 0.7812 | 72.45% | 70.18% |
| XGBoost | 0.8023 | 74.56% | 72.89% |
| **LightGBM** | **0.8156** | **76.23%** | **74.12%** |

**Best Model:** LightGBM with ROC-AUC of 0.8156

## Figures Generated
The notebook automatically saves 12 visualization figures:
- Class distribution (before/after SMOTE)
- Missing values heatmap
- ROC & Precision-Recall curves
- Confusion matrices for all models
- SHAP summary and bar plots
- Differential privacy trade-off graph

## How to Run
1. Open the notebook in Google Colab
2. Run all cells sequentially
3. Upload `application_train.csv` when prompted
4. Wait ~5-10 minutes for training

## Requirements
- Python 3.8+
- Libraries: lightgbm, xgboost, shap, imbalanced-learn, diffprivlib, scikit-learn

## Key Findings
- Credit income ratio and external scores are the strongest predictors
- LightGBM outperforms XGBoost by ~1.3% in ROC-AUC
- Differential privacy with ε=5.0 maintains 90% of baseline accuracy

## Author
[Your Name]
[Your LinkedIn/GitHub]
