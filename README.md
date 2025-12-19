# 💳 Fraud Detection Web Application

This project is an end-to-end **machine learning fraud detection system** built using real banking transaction data.  
It predicts the **probability of a transaction being fraudulent** based on transaction amount, user behavior, and historical patterns.

The entire model training was done using a **Kaggle Notebook**, and the final trained model was deployed as a **Streamlit web application**.

---

## 📌 Problem Statement

Banking fraud datasets are **highly imbalanced** — only a very small percentage of transactions are fraudulent.  
The goal of this project is to **identify risky transactions and provide a fraud probability score**, instead of just a yes/no output.

---
## 🔗 Live Demo

🚀 Deployed using **Streamlit Cloud**

👉 https://federated-learning-based-fraud-detection-in-financial-networks.streamlit.app/

Users can interact with the application by entering transaction details and instantly receive a fraud risk probability.
---

## 📊 Dataset

- Source: **NIBSS Nigerian Bank Fraud Dataset (Kaggle)**
- Total records: **1,000,000 transactions**
- Fraud rate: **~0.3%**
- Dataset contains transactional, behavioral, and time-based features

---

## 🧹 Data Cleaning & Column Reduction

The original dataset had **38 columns**, many of which were not useful for prediction.

### ❌ Removed unnecessary columns:
- `transaction_id`
- `customer_id`
- `timestamp`
- `fraud_technique`

These columns were removed because they:
- Do not contribute to prediction
- Cause data leakage
- Are not available during real-time prediction

After cleaning and encoding, the final dataset contained **59 meaningful features**.

---

## ⚙️ Data Preprocessing

### 1️⃣ Categorical Encoding
- One-hot encoded:
  - Bank
  - Channel
  - Merchant category
  - Age group
- Label encoded:
  - Location

### 2️⃣ Feature Engineering
- Time-based features (hour, day, month)
- Cyclic encoding (sin/cos for time)
- Transaction behavior features:
  - Amount log
  - Velocity score
  - Risk scores

### 3️⃣ Scaling
- Applied **StandardScaler** to numerical features
- Same scaler reused during deployment to ensure consistency

---

## ⚖️ Handling Class Imbalance

Since fraud transactions are extremely rare:
- Used **class weights**
- Used **scale_pos_weight** in XGBoost
- Evaluated models using **Recall, F1-score, ROC-AUC**, not just accuracy

---

## 🤖 Model Training (Kaggle Notebook)

Three models were trained and compared in a Kaggle notebook:

| Model | Observation |
|------|------------|
| Logistic Regression | High recall but too many false positives |
| Random Forest | Strong baseline performance |
| **XGBoost** | Best balance of precision, recall, and ROC-AUC |

📌 **XGBoost performed best and was selected as the final model.**

---

## ✅ Final Model Used

- **Model:** XGBoost Classifier
- Trained with imbalance-aware parameters
- Outputs **fraud probability** instead of binary result

### Saved artifacts:
- `fraud_detection_xgb.pkl` / `fraud_detection_xgb.json`
- `scaler.pkl`
- `label_encoder_location.pkl`
- `feature_list.json`

---

## 🌐 Web Application (Streamlit)

The trained model is deployed using **Streamlit**.

### App Features:
- User-friendly input form
- Real-time fraud probability prediction
- Risk categorization:
  - ✅ Low Risk
  - ⚠️ Medium Risk
  - 🚨 High Risk

The app reconstructs the **exact feature order used during training**, applies scaling, and then performs prediction.

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Kaggle Notebook (for training)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/your-username/fraud-detection-app.git
cd fraud-detection-app

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt
streamlit run app.py


