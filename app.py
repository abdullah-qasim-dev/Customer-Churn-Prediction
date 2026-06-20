import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Prediction")
st.markdown("Enter customer details below to predict whether they will churn or not.")
st.markdown("---")

# ── INPUT FIELDS ──────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.markdown("---")

# ── ENCODE INPUTS ─────────────────────────────────────────

def encode_inputs():
    data = {
        "gender":           1 if gender == "Male" else 0,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "Partner":          1 if partner == "Yes" else 0,
        "Dependents":       1 if dependents == "Yes" else 0,
        "tenure":           tenure,
        "PhoneService":     1 if phone_service == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,

        # MultipleLines
        "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "MultipleLines_Yes":              1 if multiple_lines == "Yes" else 0,

        # InternetService
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No":          1 if internet_service == "No" else 0,

        # OnlineSecurity
        "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
        "OnlineSecurity_Yes":                 1 if online_security == "Yes" else 0,

        # OnlineBackup
        "OnlineBackup_No internet service": 1 if online_backup == "No internet service" else 0,
        "OnlineBackup_Yes":                 1 if online_backup == "Yes" else 0,

        # DeviceProtection
        "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
        "DeviceProtection_Yes":                 1 if device_protection == "Yes" else 0,

        # TechSupport
        "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
        "TechSupport_Yes":                 1 if tech_support == "Yes" else 0,

        # StreamingTV
        "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
        "StreamingTV_Yes":                 1 if streaming_tv == "Yes" else 0,

        # StreamingMovies
        "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
        "StreamingMovies_Yes":                 1 if streaming_movies == "Yes" else 0,

        # Contract
        "Contract_One year":  1 if contract == "One year" else 0,
        "Contract_Two year":  1 if contract == "Two year" else 0,

        # PaymentMethod
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check":        1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check":            1 if payment_method == "Mailed check" else 0,
    }
    return pd.DataFrame([data])

# ── PREDICT BUTTON ────────────────────────────────────────

if st.button("🔍 Predict Churn", use_container_width=True):
    input_df = encode_inputs()
    # input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to **CHURN**")
    else:
        st.success(f"✅ This customer is **NOT likely to Churn**")

    # st.metric(label="Churn Probability", value=f"{probability * 100:.1f}%")

    # # Probability bar
    # st.progress(float(probability))

    # if probability >= 0.7:
    #     st.warning("🔴 High Risk — Immediate retention action recommended!")
    # elif probability >= 0.4:
    #     st.warning("🟡 Medium Risk — Monitor this customer closely.")
    # else:
    #     st.info("🟢 Low Risk — Customer appears stable.")