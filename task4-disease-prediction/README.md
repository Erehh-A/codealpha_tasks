# Disease Prediction from Medical Data

## Objective
Predict the possibility of disease (breast cancer) based on structured patient diagnostic data, using classification techniques.

## Dataset
Breast Cancer Wisconsin (Diagnostic) dataset, loaded via `sklearn.datasets.load_breast_cancer()`.
569 records, 30 numeric features derived from digitized images of breast mass cell nuclei (e.g. radius, texture, perimeter, concavity, symmetry — each reported as mean, standard error, and "worst" value).

Target variable: presence of malignant tumor.
- `1` = malignant (212 cases)
- `0` = benign (357 cases)

*(Note: the raw sklearn dataset encodes this in the opposite direction — 0=malignant, 1=benign — and was flipped here so `1` consistently represents "the condition we need to catch," matching standard classification convention.)*

## Approach

1. **EDA**
   - Verified no missing values and fully numeric features (no encoding required).
   - Checked class balance (~63% benign / 37% malignant).
   - Built a correlation heatmap, identifying strong multicollinearity among size-related features (`radius`, `perimeter`, `area` — all mathematically related measurements of tumor size).
   - Identified `worst concave points`, `worst perimeter`, and `worst radius` as the features most strongly correlated with malignancy, consistent with medical intuition: malignant tumors tend to be larger with more irregular, concave borders.
2. **Preprocessing**
   - Stratified 80/20 train-test split to preserve class balance.
   - Scaled all features with `StandardScaler` (important for SVM in particular, given the wide differences in feature scale).
3. **Modeling** — trained and compared four classifiers:
   - Logistic Regression
   - Random Forest
   - SVM
   - XGBoost
4. **Evaluation** — used Accuracy, Precision, Recall, F1-Score, and ROC-AUC, prioritizing **Recall on the malignant class**, since in a medical context, a missed malignant case (false negative) is far more costly than a false alarm.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| **Random Forest** | **0.974** | **1.000** | 0.929 | **0.963** | 0.993 |
| SVM | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 |
| XGBoost | 0.965 | 1.000 | 0.905 | 0.950 | 0.990 |

**Final model: Random Forest.** It achieved perfect Precision (no false alarms) and matched the best Recall among all models (catching ~93% of malignant cases), while also offering interpretable feature importances — a practical advantage over SVM, which tied it on performance but is harder to explain.

## Key Notes

- All models performed strongly (96%+ accuracy, 0.99+ ROC-AUC), reflecting that this dataset's features separate the two classes cleanly.
- XGBoost, despite being a powerful algorithm generally, slightly underperformed the simpler models here — a reminder that more complex models don't always win on small, clean datasets.
- Multicollinearity among size-based features was identified via the correlation heatmap but not removed, since both Random Forest and XGBoost are naturally robust to it.

## Project Structure
```
disease-prediction-model/
├── notebooks/
│   └── 01_eda.ipynb
├── models/
│   ├── disease_prediction_rf.pkl
│   └── scaler.pkl
└── README.md
```

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

## Possible Next Steps
- Feature importance analysis to identify top diagnostic predictors
- Hyperparameter tuning (GridSearchCV) for Random Forest / XGBoost
- Deploy as a simple API or Streamlit app for interactive predictions