import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="??", layout="wide")

st.title("?? Customer Churn Prediction System")
st.markdown("Predict whether a customer is **Likely to Stay** or **Likely to Churn** using Machine Learning.")

@st.cache_resource
def load_bundle():
    model_path = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")
    return joblib.load(model_path)

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (Months with company)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=65.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(monthly_charges * tenure))

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

if st.button("?? Run Churn Prediction", type="primary"):
    raw_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    input_df = pd.DataFrame([raw_data])
    encoded_df = pd.get_dummies(input_df)
    
    aligned_df = pd.DataFrame(0, index=[0], columns=feature_names)
    for col in encoded_df.columns:
        if col in aligned_df.columns:
            aligned_df[col] = encoded_df[col]
            
    scaled_input = scaler.transform(aligned_df)
    prob_churn = float(model.predict_proba(scaled_input)[0, 1])
    prob_stay = float(1.0 - prob_churn)
    
    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    
    with col1:
        if prob_churn >= 0.5:
            st.error(f"?? **High Churn Risk!**\n\nProbability of leaving: **{prob_churn*100:.1f}%**")
        else:
            st.success(f"? **Customer Likely to Stay!**\n\nProbability of staying: **{prob_stay*100:.1f}%**")
            
    with col2:
        st.metric("Churn Probability", f"{prob_churn*100:.1f}%")
        st.metric("Retention Probability", f"{prob_stay*100:.1f}%")
