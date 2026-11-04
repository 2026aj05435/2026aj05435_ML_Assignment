# Employee Attrition Prediction

## a. Problem Statement

This project predicts whether an employee is likely to leave the company (attrition) based on their job, pay, satisfaction and work-history details. Employee attrition is costly: every resignation means hiring and training a replacement, losing experience and disrupting teams. Five classification models were trained and compared on the same dataset, and the results are presented in an interactive Streamlit app where HR users can upload employee data and evaluate each model.

## b. Dataset Description

The **IBM HR Analytics Employee Attrition & Performance** dataset is a fictional HR dataset created by IBM data scientists.

- **Source:** [Kaggle – IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Size:** 1,470 employees (rows) × 35 columns
- **Target:** `Attrition`, Yes = employee left, No = employee stayed (binary classification)
- **Class balance:** 237 Yes (16%) vs 1,233 No (84%), an imbalanced dataset
- **Features used:** 30 (23 numeric, 7 categorical). Four columns were removed because they carry no information: `EmployeeNumber` (an ID), and `EmployeeCount`, `Over18` and `StandardHours` (the same value in every row)
- **Missing values:** none
- **Preprocessing:** numeric features scaled with `StandardScaler`, categorical features one-hot encoded with `OneHotEncoder`, both inside a scikit-learn `Pipeline` to avoid data leakage
- **Train/test split:** 80/20, stratified on the target (1,176 training rows, 294 test rows)

## c. GitHub Repository Link

https://github.com/2026aj05435/2026aj05435_ML_Assignment

**Live Streamlit app:** [ADD YOUR STREAMLIT LINK HERE]

## d. Models Used

All five models were trained on the same training data and evaluated on the same 294-row test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8605 | 0.8115 | 0.6154 | 0.3404 | 0.4384 | 0.3871 |
| Decision Tree | 0.7653 | 0.6105 | 0.3103 | 0.3830 | 0.3429 | 0.2036 |
| KNN | 0.8435 | 0.5946 | 0.5385 | 0.1489 | 0.2333 | 0.2222 |
| Naive Bayes | 0.6463 | 0.7032 | 0.2605 | 0.6596 | 0.3735 | 0.2265 |
| Support Vector Machine | 0.8741 | 0.8162 | 0.7083 | 0.3617 | 0.4789 | 0.4462 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best model overall (AUC 0.81, MCC 0.39), close to SVM despite being a simple linear model, which suggests the attrition patterns are largely linear. Its weakness is recall: it caught only 16 of the 47 leavers (0.34), because at the default 50% threshold it favours the majority class "stays". The good AUC shows it ranks leavers sensibly, so a lower threshold could catch more of them. |
| Decision Tree | Weakest model on MCC (0.20) and AUC (0.61). It caught 18 of 47 leavers but raised 40 false alarms (precision 0.31). With no depth limit, the tree kept splitting until it memorised the training data (overfitting), so it generalises poorly to new employees. Its hard 0/1 leaf outputs also give poor probability ranking, which explains the low AUC. |
| KNN | Accuracy (0.84) looks reasonable but equals the "always predict stays" baseline. Recall is only 0.15 (7 of 47 leavers caught) and AUC is 0.59, close to random guessing. After one-hot encoding there are about 50 features, so distances between employees become less meaningful, and because 84% of neighbours are stayers, the majority vote almost always predicts "stays". |
| Naive Bayes | Highest recall of all models (0.66), catching 31 of 47 leavers. The cost is very low precision (0.26): 88 employees were wrongly flagged, giving the lowest accuracy (0.65). It assumes all features are independent, but features such as JobLevel, MonthlyIncome and TotalWorkingYears are strongly related, which distorts its probabilities. It suits a "flag widely, review manually" approach. |
| Support Vector Machine | Best model on 5 of 6 metrics: accuracy 0.87, AUC 0.82, precision 0.71, F1 0.48 and MCC 0.45. When it flags an employee it is right 7 times out of 10 (17 of 24), making its alerts trustworthy. Like the other models, its recall is limited (0.36, 17 of 47 leavers), because the class imbalance pushes predictions towards "stays". |
| **Overall winner** | **Support Vector Machine.** Because only 16% of employees left, accuracy is misleading: predicting "stays" for everyone already scores 84%. Models were therefore compared mainly on MCC and F1, which account for both classes. SVM leads on both (MCC 0.45, F1 0.48), with Logistic Regression a close second. If catching every possible leaver mattered more than false alarms, Naive Bayes's high recall would be worth considering. |

## Repository Structure

```
├── app.py                       # Streamlit app
├── hr_attrition_models.ipynb    # Data preparation, training and evaluation
├── requirements.txt             # Python dependencies
├── test_data.csv                # Test set (294 rows) used in the app
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Original dataset
└── model/                       # Saved trained pipelines (.joblib)
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501, choose a model in the sidebar and upload `test_data.csv`.