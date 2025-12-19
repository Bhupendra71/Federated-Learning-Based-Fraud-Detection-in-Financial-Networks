import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb

# ===========================
# Load saved ML artifacts
# ===========================
booster = xgb.Booster()
booster.load_model("fraud_detection_xgb.json")

scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder_location.pkl")

with open("feature_list.json", "r") as f:
    FEATURE_LIST = json.load(f)

# ===========================
# Streamlit Page Config
# ===========================
st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("💳 Banking Fraud Detection")
st.write("Enter transaction details to predict fraud probability")

# ===========================
# User Inputs
# ===========================
amount = st.number_input("Transaction Amount (in ₦ )  ie.1000₦1 (Nigerian Naira) ≈ ₹150", min_value=0.0, step=100.0)
location = st.selectbox("Location", label_encoder.classes_)
hour = st.slider("Hour of Transaction", 0, 23, 12)
day_of_week = st.slider("Day of Week (0 = Monday, 6 = Sunday)", 0, 6, 3)
month = st.slider("Month (1 = Jan, 12 = Dec)", 1, 12, 6)

bank = st.selectbox("Bank", ["Access", "Fidelity", "GTBank", "UBA", "Zenith", "FirstBank", "Wema", "Sterling", "Union", "FCMB"])
channel = st.selectbox("Channel", ["Mobile", "Web", "POS"])
merchant_category = st.selectbox("Merchant Category", ["Grocery", "Entertainment", "Transport", "Education"])
age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-50", "50+"])

# ===========================
# Prediction Logic
# ===========================
if st.button("Predict Fraud Risk"):

    df_input = pd.DataFrame(
        np.zeros((1, len(FEATURE_LIST))),
        columns=FEATURE_LIST
    )

    df_input["amount"] = amount
    df_input["hour"] = hour
    df_input["day_of_week"] = day_of_week
    df_input["month"] = month
    df_input["location"] = label_encoder.transform([location])[0]

    for col in [
        f"bank_{bank}",
        f"channel_{channel}",
        f"merchant_category_{merchant_category}",
        f"age_group_{age_group}"
    ]:
        if col in df_input.columns:
            df_input[col] = 1

    numeric_cols = scaler.feature_names_in_
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

    dmat = xgb.DMatrix(df_input)
    fraud_prob = booster.predict(dmat)[0]

    st.subheader("🔍 Prediction Result")
    st.metric("Fraud Probability", f"{fraud_prob * 100:.3f}%")

    if fraud_prob >= 0.85:
        st.error("🚨 Very High Fraud Risk")
    elif fraud_prob >= 0.6:
        st.warning("⚠️ High Fraud Risk")
    elif fraud_prob >= 0.3:
        st.warning("🟡 Medium Fraud Risk")
    else:
        st.success("✅ Low Fraud Risk")

