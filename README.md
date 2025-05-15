# Handling Missing Values – Machine Learning Impact Study 🤖📉

This repository contains an exploratory analysis of different techniques for handling missing data, using the **Pima Indians Diabetes Dataset**. The study evaluates how these strategies affect the performance of several machine learning models in terms of F1-score and ROC AUC. The goal is to determine optimal approaches for imputation based on the patterns and mechanisms of missingness within the dataset.

## 📁 Dataset

The dataset used is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). It includes key medical predictor variables such as `Glucose`, `Diastolic_BP`, `Skin_Fold`, `Serum_Insulin`, `BMI`, among others. Several of these features have missing values, particularly `Serum_Insulin` and `Skin_Fold`, which present substantial data loss (374 and 227 missing entries, respectively).

## 📌 Objectives

- Explore the extent and distribution of missing values
- Analyze the impact of different imputation techniques:
  - Row deletion
  - Mean imputation
  - Median imputation
  - Column deletion
  - KNN Imputer
  - MICE (Multiple Imputation by Chained Equations)
- Evaluate model performance across strategies using:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Dummy Classifier (baseline)
- Determine which imputation strategies are most effective under various missingness patterns (MCAR, MAR, MNAR)

## 📊 Main Findings

- The highest correlation was found between `Serum_Insulin` and `Skin_Fold`, suggesting multivariate imputation techniques might be beneficial.
- KS test p-values indicated `Skin_Fold`, `Serum_Insulin`, and `BMI` exhibit **MAR** patterns; `Glucose` is likely **MCAR**.
- **Row deletion** significantly reduced dataset size and yielded suboptimal performance, especially for XGBoost.
- **Mean/Median Imputation** performed well, especially when combined with removal of columns with over 20% missingness (`Skin_Fold` and `Serum_Insulin`). Median outperformed mean on skewed variables.
- **KNN Imputer** maintained competitive ROC AUC values but showed reduced F1-scores, likely due to noise or over-smoothing.
- **MICE** provided the highest F1 for Logistic Regression but showed variable results across models, sometimes underperforming.

## 💡 Conclusion

The best-performing strategy was the combination of removing high-missingness columns (`Skin_Fold`, `Serum_Insulin`) with **median imputation** on the remaining data. This approach offered the highest balance in F1-score and ROC AUC across models, especially Logistic Regression and XGBoost.

Simple imputation methods often matched or exceeded the performance of advanced methods like KNN and MICE, particularly in scenarios where simplicity and robustness are priorities. However, MICE and KNN remain valuable for capturing multivariate dependencies, especially under MAR assumptions.

Ultimately, no single imputation strategy is universally optimal. Each dataset’s structure and missingness mechanism must guide the approach. Future work should explore more context-aware and domain-informed imputers, and validate conclusions on external datasets to ensure generalizability and fairness in healthcare applications.
