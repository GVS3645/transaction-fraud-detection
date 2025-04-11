# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from api.fraud.Fraud import Fraud

# Load the trained model
model = joblib.load('models/model_cycle1.joblib')

# Initialize the Fraud pipeline
pipeline = Fraud()

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("🔍 Transaction Fraud Detection")
st.markdown("Enter transaction details below to detect possible fraud.")

# Input fields
type_input = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "CASH_IN", "DEBIT", "PAYMENT"])
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, format="%.2f")
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, format="%.2f")
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, format="%.2f")

if st.button("Predict Fraud"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'type': type_input,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest
    }])

    # Pipeline steps
    cleaned_data = pipeline.data_cleaning(input_data)
    features = pipeline.feature_engineering(cleaned_data)
    prepared_data = pipeline.data_preparation(features)

    # Get prediction
    result_df = pipeline.get_prediction(model, input_data, prepared_data)

    # Show result
    prediction = result_df['prediction'][0]
    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction appears to be legitimate.")
