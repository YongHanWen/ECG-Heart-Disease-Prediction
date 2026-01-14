# ECG Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease using **non-invasive, ECG-related clinical features**.


---


## 🩺 Problem Overview
Heart disease diagnosis often requires invasive tests.  
This project explores whether **ECG-derived clinical values** can be used to accurately predict heart disease risk.


---


## 📊 Dataset
- 1,190 patient records
- 11 clinical features, including:
  - ChestPainType
  - ExerciseAngina
  - ST_Slope
  - RestingBP
  - MaxHeartRate
  - Cholesterol

### Data Cleaning Performed
- Converted text fields → categorical
- Removed invalid values
- Checked duplicates (none found)
- Handled outliers and skewed distributions

---

## 🔍 Exploratory Data Analysis
- Identified key feature relationships
- Used **Cramer’s V** to measure strength of association
- Found top predictors:
  - **ST_Slope**
  - **ChestPainType**
  - **ExerciseAngina**

---

## 🤖 Machine Learning Models
Models trained & evaluated:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- Soft Voting Ensemble  

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- ROC AUC
- Confusion Matrix

---

## 🏆 Model Performance Summary
- **XGBoost and Random Forest performed best overall**
- Logistic Regression gave solid performance with low complexity
- Ensemble model was stable but did not outperform the best single models

---

## 🎮 Demo Application
A simple app script allows users to input patient data and returns:
- Prediction (Heart Disease: Yes/No)
- Risk Level (Low / Medium / High)

---

## 📁 Repository Structure
```text
├── ecg_heart_disease_prediction.ipynb
├── heart_disease_model.pkl
├── Heart Failure prediction datasets.csv
├── app.py
├── images/
├── templates/
└── docs / slides
```

---

## 🧠 Conclusion
ECG-based features can provide meaningful early indicators of heart disease, showing strong alignment between clinical intuition, EDA findings, and model results.

---

