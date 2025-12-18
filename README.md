# 💳 Fraud Detection Web Application

This project is an end-to-end machine learning system that predicts the probability of a banking transaction being fraudulent based on user-provided transaction details.

The goal of this project was to understand how real-world fraud detection systems are built — from data preprocessing and model training to deployment as an interactive web application.

---

## 🚀 Features

- Real-time fraud probability prediction
- Handles highly imbalanced fraud data
- Advanced feature engineering
- Risk-based decision thresholds
- Clean and simple web interface

---

## 🧠 Machine Learning Models

I trained and compared three models:

- Logistic Regression
- Random Forest
- **XGBoost (final model used)**

XGBoost was selected because it achieved the best ROC-AUC score and recall for fraud cases while handling class imbalance effectively.

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit

---

## 📊 Dataset

- Source: Nigerian Banking Fraud Dataset (NIBSS)
- Size: ~1 million transactions
- Fraud ratio: ~0.3%

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/your-username/fraud-detection-app.git
cd fraud-detection-app

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

📌 Project Learnings

Handling extreme class imbalance

Feature engineering for transactional data

Probability calibration and risk thresholds

End-to-end ML deployment

📌 Project Learnings

Handling extreme class imbalance

Feature engineering for transactional data

Probability calibration and risk thresholds

End-to-end ML deployment

---
