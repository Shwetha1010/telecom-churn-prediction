import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit App Title
st.title("📊 Telecom Customer Churn Prediction App")
st.write("Predict whether a telecom customer is likely to churn based on their account and service usage details.")

st.sidebar.header("📋 Enter Customer Details")

def user_input():
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 2000.0)

    data = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone == "Yes" else 0,
        'MultipleLines': {"No": 0, "Yes": 1, "No phone service": 2}[multiple],
        'InternetService': {"DSL": 0, "Fiber optic": 1, "No": 2}[internet],
        'OnlineSecurity': {"No": 0, "Yes": 1, "No internet service": 2}[security],
        'OnlineBackup': {"No": 0, "Yes": 1, "No internet service": 2}[backup],
        'DeviceProtection': {"No": 0, "Yes": 1, "No internet service": 2}[protection],
        'TechSupport': {"No": 0, "Yes": 1, "No internet service": 2}[support],
        'StreamingTV': {"No": 0, "Yes": 1, "No internet service": 2}[tv],
        'StreamingMovies': {"No": 0, "Yes": 1, "No internet service": 2}[movies],
        'Contract': {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'PaymentMethod': {
            "Electronic check": 0,
            "Mailed check": 1,
            "Bank transfer (automatic)": 2,
            "Credit card (automatic)": 3
        }[payment],
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    return pd.DataFrame([data])

# Get user input
input_df = user_input()

# Ensure column order matches model training
expected_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

input_df = input_df[expected_columns]

# Scale numerical features
input_scaled = scaler.transform(input_df)

# Predict button
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to churn**.\n\n💡 Probability: **{probability:.2f}**")
    else:
        st.success(f"✅ The customer is **likely to stay**.\n\n💡 Probability: **{probability:.2f}**")
