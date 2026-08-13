# CodeAlpha Machine Learning Internship — Tasks

This repository contains my submissions for the CodeAlpha Machine Learning Internship. Each task is organized in its own folder with a dedicated notebook, saved model, and README explaining the approach, results, and reasoning.

## Tasks Completed

### ✅ Task 1: Credit Scoring Model
Predicts an individual's creditworthiness (good/bad credit risk) using the German Credit Risk dataset. Compares Logistic Regression, Decision Tree, and Random Forest, with a focus on Recall for the "bad risk" class given the real-world cost of missing risky loans.

📂 [`/task1-credit-scoring`](./task1-credit-scoring) — [Read the full writeup](./task1-credit-scoring/README.md)

**Best model:** Logistic Regression (class-weighted) — Recall 0.72, ROC-AUC 0.76

---

### ✅ Task 4: Disease Prediction from Medical Data
Predicts the likelihood of malignant breast cancer using the Breast Cancer Wisconsin dataset. Compares Logistic Regression, Random Forest, SVM, and XGBoost, prioritizing Recall on the malignant class given the cost of a missed diagnosis.

📂 [`/task4-disease-prediction`](./task4-disease-prediction) — [Read the full writeup](./task4-disease-prediction/README.md)

**Best model:** Random Forest — Precision 1.00, Recall 0.93, ROC-AUC 0.993

---

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Jupyter Notebook

## Repository Structure
```
codealpha_tasks/
├── task1-credit-scoring/
│   ├── data/
│   ├── notebooks/
│   ├── models/
│   └── README.md
├── task4-disease-prediction/
│   ├── notebooks/
│   ├── models/
│   └── README.md
└── README.md   (this file)
```

## About
Completed as part of the CodeAlpha Machine Learning Internship. Each task README includes the objective, approach, evaluation metrics, and key insights drawn from the analysis.