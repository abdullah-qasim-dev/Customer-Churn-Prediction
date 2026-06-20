# Customer Churn Prediction Using Machine Learning

A complete end-to-end machine learning pipeline that predicts whether a telecom customer will churn (leave the service), built using the IBM Telco Customer Churn dataset. The project includes exploratory data analysis, preprocessing, multiple model comparisons, and a deployed interactive web app using Streamlit.

## Overview

Customer churn is a critical problem in the telecom industry, where retaining an existing customer is significantly cheaper than acquiring a new one. This project builds a machine learning pipeline to identify customers likely to churn, enabling proactive retention strategies. The pipeline covers EDA, preprocessing, model training (with class imbalance handling), hyperparameter tuning, evaluation, and deployment via a Streamlit web app.

## Dataset

- **Source:** [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Features:** 21 columns (demographics, account info, subscribed services)
- **Target:** `Churn` (Yes/No)
- **Class distribution:** 73.46% No Churn vs. 26.54% Churn (imbalanced)
- **Missing values:** 11 (in `TotalCharges`, handled via median imputation)
- **Duplicates:** None found

### Feature Categories

- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Account Info:** tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

## Exploratory Data Analysis — Key Insights

- **Contract type** is one of the strongest churn predictors — month-to-month customers churn far more than one/two-year contract customers.
- **Monthly charges** are higher among churned customers (median ~$80) vs. non-churned (~$65).
- **Tenure** shows a U-shaped distribution — many new customers and many long-term loyal ones, with mid-tenure customers at higher churn risk.
- `tenure` and `TotalCharges` are strongly correlated (0.83), as expected.

## Data Preprocessing

- Converted `TotalCharges` from object to numeric (whitespace entries caused type issues)
- Imputed 11 missing values using the median
- Dropped `customerID` (non-predictive identifier)
- Label-encoded binary columns: `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`
- One-hot encoded multi-class categorical columns (`drop_first=True`) — expanded dataset from 21 to 31 columns
- No outliers removed (high `TotalCharges` values reflect genuine long-tenure customers)
- 80/20 train-test split (`random_state=42`), features scaled using `StandardScaler` (fit on train only, to avoid data leakage)

## Models Trained

| Model | Variant |
|-------|---------|
| Logistic Regression | Standard + Class-Balanced |
| Decision Tree | Standard + Class-Balanced (tuned: max_depth=5) |
| Random Forest | Standard + Class-Balanced (tuned: max_depth=6) + RandomizedSearchCV-optimized |

Class imbalance was handled using `class_weight='balanced'`, giving more weight to the minority (churn) class. `RandomizedSearchCV` (5-fold CV, F1-optimized, 20 iterations) was used to find the best Random Forest configuration:

- `n_estimators`: 200
- `max_depth`: 10
- `min_samples_split`: 2
- `min_samples_leaf`: 4
- `class_weight`: balanced

## Results

| Model | Accuracy | F1 (Churn) | ROC-AUC | Recall (Churn) |
|-------|----------|------------|---------|----------------|
| Logistic Regression | 0.820 | 0.636 | 0.862 | 0.60 |
| LR (Balanced) | 0.747 | 0.633 | 0.862 | 0.82 |
| Decision Tree (Tuned) | 0.808 | 0.559 | 0.850 | 0.46 |
| DT (Balanced) | 0.745 | 0.622 | 0.836 | 0.79 |
| Random Forest (Tuned) | 0.805 | 0.568 | 0.866 | 0.49 |
| **RF (Balanced)** ⭐ | **0.755** | **0.648** | **0.866** | **0.85** |
| RF (RandomizedSearchCV) | 0.785 | 0.660 | 0.865 | 0.79 |

### Best Model: Random Forest (Balanced)

Selected as the best model due to its **highest recall (0.85)** for the churn class — correctly identifying 317 out of 373 churners, missing only 56. In churn prediction, **recall is prioritized over accuracy**, since failing to catch a real churner leads to direct revenue loss, while flagging a loyal customer (lower precision) only costs a retention offer.

**Classification Report — RF (Balanced):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Churn (0) | 0.93 | 0.72 | 0.81 | 1036 |
| Churn (1) | 0.52 | 0.85 | 0.65 | 373 |

- **5-Fold CV F1 Score:** 0.624 (± 0.017) — consistent performance across folds
- **ROC-AUC:** 0.866 — strong discriminative ability

### Top Churn Drivers (Feature Importance)

1. `tenure`
2. `Contract_Two year`
3. `TotalCharges`
4. `InternetService_Fiber optic`
5. `PaymentMethod_Electronic check`
6. `MonthlyCharges`
7. `Contract_One year`
8. `OnlineSecurity_Yes`

**Business takeaway:** Month-to-month customers with high monthly charges and low tenure represent the highest churn risk — ideal targets for retention offers like contract upgrades or loyalty discounts.

## Streamlit Web Application

An interactive web app was built to provide real-time churn predictions using the trained RF (Balanced) model and the fitted `StandardScaler`.

**Features:**
- Interactive form with dropdowns/sliders for all customer features
- Real-time churn / no-churn prediction
- Risk level classification (High / Medium / Low) based on predicted probability
- Clean two-column layout

**Workflow:** User input → encoded (same logic as training) → scaled with saved `StandardScaler` → passed to the trained model → prediction + risk level displayed.

## How to Run

### 1. Train the model (generates `churn_model.pkl` and `scaler.pkl`)

Run the main notebook/script (`complete_code`) end-to-end. This performs preprocessing, training, evaluation, and saves the trained model and scaler.

### 2. Run the Streamlit app

```bash
pip install streamlit joblib scikit-learn pandas numpy
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

> **Note:** `churn_model.pkl` and `scaler.pkl` are generated artifacts and are not included in this repository. Running the training code will regenerate them automatically before launching the app.

## Project Structure

```
├── complete_code(.ipynb/.py)        # Full pipeline: EDA, preprocessing, training, evaluation
├── app.py                           # Streamlit web application
├── Telco-Customer-Churn.csv         # Raw dataset
├── telco_churn_preprocessed.csv     # Preprocessed dataset (optional)
├── Visualizations/                  # Generated charts and plots
├── Project_Report.pdf               # Full project report
└── README.md
```

## Future Work

- Implement advanced ensemble methods (XGBoost, LightGBM)
- Experiment with SMOTE for class imbalance handling
- Incorporate time-series analysis of churn trends
- Deploy the app on a cloud platform (Streamlit Cloud, Heroku)

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib

## References

- IBM. (2019). Telco Customer Churn Dataset. [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
