# Credit Scoring Model
 
## Objective
Predict an individual's creditworthiness (good vs. bad credit risk) using past financial and demographic data, to help automate and support lending decisions.
 
## Dataset
German Credit Risk dataset (1000 records, 10 features + target).
Source: Kaggle — [German Credit Risk - With Target](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)
 
Target variable: `Risk` (good / bad), with a 70/30 class split.
 
**Note on features:** while the dataset does not include explicit income or payment history fields, `Job`, `Housing`, `Saving accounts`, and `Checking account` serve as reasonable proxies for financial standing and history, which is consistent with how credit risk is assessed using categorical banking data.
 
## Approach
 
1. **EDA** — explored distributions of Age, Credit Amount, and Duration; identified right-skew in financial features; visualized relationships between Duration, Credit Amount, Age, and Risk using boxplots.
2. **Preprocessing**
   - Filled missing values in `Saving accounts` and `Checking account` with a `"none"` category (missing indicates the applicant doesn't hold that account type, not an unknown value).
   - Label-encoded binary categorical features (`Sex`, `Housing`).
   - One-hot encoded multi-category features (`Saving accounts`, `Checking account`, `Purpose`).
   - Scaled numeric features using `StandardScaler`.
   - Stratified 80/20 train-test split to preserve class balance.
3. **Modeling** — trained and compared three classifiers:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   Each was evaluated both with default settings and with `class_weight='balanced'` to address the 70/30 class imbalance.
4. **Evaluation** — used Accuracy, Precision, Recall, F1-Score, and ROC-AUC, prioritizing **Recall on the "bad" risk class**, since in credit scoring, failing to catch a risky loan is more costly than being overly cautious with a safe one.
## Results
 
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (balanced) | 0.705 | 0.51 | **0.72** | 0.59 | **0.76** |
| Decision Tree (balanced) | 0.720 | 0.54 | 0.47 | 0.50 | 0.65 |
| Random Forest (balanced) | 0.750 | 0.65 | 0.37 | 0.47 | 0.75 |
 
**Final model: Logistic Regression with `class_weight='balanced'`**, chosen for its strongest Recall and ROC-AUC — it catches significantly more actual risky loans than the alternatives, which matters more than raw accuracy in this context.
 
## Key Insights (from model coefficients)
 
- Loan **Duration** is positively associated with risk — longer loans are riskier, consistent with EDA findings.
- **Purpose of loan** matters: education and repairs loans carry higher predicted risk than car or radio/TV purchases.
- Having savings marked as **"rich"** is one of the strongest indicators of lower risk.
- Surprisingly, raw **Credit Amount** and **Age** had weak influence on the final model once other features were accounted for, despite visible trends in EDA — a reminder that univariate visual trends don't always translate directly into strong model weights.
## Project Structure
```
credit-scoring-model/
├── data/
│   └── german_credit_data.csv
├── notebooks/
│   └── main.ipynb
├── models/
│   ├── credit_scoring_logreg.pkl
│   └── scaler.pkl
└── README.md
```
 
## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
 
## Possible Next Steps
- Hyperparameter tuning (GridSearchCV) for Random Forest / Decision Tree
- Try SMOTE for oversampling instead of class weighting
- Deploy as a simple API or Streamlit app for interactive predictions
 