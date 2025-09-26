# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="📉 Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# Sidebar: Modern Design
# ------------------------------
st.sidebar.markdown("## 🚀 Customer Churn Predictor")
st.sidebar.markdown(
    """
    Welcome! This app predicts whether a telecom customer will **churn** or **stay**.
    
    ### ⚡ Features
    - Predict single customer churn 🧑
    - Batch prediction from CSV 📄
    - Interactive sliders & number inputs 🔢
    - Probability output with charts 📊
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📌 Quick Links")
st.sidebar.markdown(
      """
    - [LinkedIn](https://www.linkedin.com/in/mirza-yasir-abdullah-baig/) 🔗
    - [GitHub](https://github.com/mirzayasirabdullahbaig07) 💻
    - [Kaggle](https://www.kaggle.com/mirzayasirabdullah07) 🏆
    """

)
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 Tips")
st.sidebar.info(
    """
    - Use sliders to quickly adjust numeric values.  
    - Upload CSVs with the same columns as training data.  
    - Scroll down to view all predictions.  
    """
)

# ------------------------------
# Load Model and Encoders
# ------------------------------
try:
    with open("churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    feature_names = model.feature_names_in_
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# ------------------------------
# Tabs: Single & Batch Prediction
# ------------------------------
tab1, tab2 = st.tabs(["🧑 Single Customer", "📄 Batch Prediction (CSV)"])

# ------------------------------
# SINGLE CUSTOMER PREDICTION
# ------------------------------
with tab1:
    st.header("Predict Churn for Single Customer")
    with st.form("single_form"):
        st.write("Fill in the customer details:")

        col1, col2 = st.columns(2)

        with col1:
            input_data = {}
            input_data['gender'] = st.selectbox("Gender", ["Female", "Male"])
            input_data['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
            input_data['Partner'] = st.selectbox("Partner", ["Yes", "No"])
            input_data['Dependents'] = st.selectbox("Dependents", ["Yes", "No"])
            input_data['tenure'] = st.slider("Tenure (months)", 0, 100, 12, 1)
            input_data['PhoneService'] = st.selectbox("Phone Service", ["Yes", "No"])
            input_data['MultipleLines'] = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            input_data['InternetService'] = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with col2:
            input_data['OnlineSecurity'] = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            input_data['OnlineBackup'] = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            input_data['DeviceProtection'] = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            input_data['TechSupport'] = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            input_data['StreamingTV'] = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            input_data['StreamingMovies'] = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            input_data['Contract'] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            input_data['PaperlessBilling'] = st.selectbox("Paperless Billing", ["Yes", "No"])
            input_data['PaymentMethod'] = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            input_data['MonthlyCharges'] = st.number_input("Monthly Charges ($)", 0.0, 1000.0, 70.0, 1.0)
            input_data['TotalCharges'] = st.number_input("Total Charges ($)", 0.0, 50000.0, 1500.0, 10.0)

        submit_button = st.form_submit_button("Predict Churn")

    if submit_button:
        input_df = pd.DataFrame([input_data])

        # Encode categorical features
        for col, encoder in encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])

        # Align columns
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Customer WILL churn! (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"✅ Customer will NOT churn (Probability: {1-prediction_proba:.2f})")

# ------------------------------
# BATCH PREDICTION FROM CSV
# ------------------------------
with tab2:
    st.header("Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV with customer data", type=["csv"])

    if uploaded_file:
        batch_data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded CSV Preview")
        st.dataframe(batch_data.head(10), height=250)

        if st.button("Predict Churn for CSV"):
            batch_features = batch_data.copy()

            # Drop extra columns not in training
            for col in batch_features.columns:
                if col not in feature_names:
                    batch_features.drop(columns=[col], inplace=True)

            # Convert numeric columns and handle blanks
            numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            for col in numeric_cols:
                if col in batch_features.columns:
                    batch_features[col] = pd.to_numeric(batch_features[col], errors='coerce').fillna(0)

            # Encode categorical columns
            for col, encoder in encoders.items():
                if col in batch_features.columns:
                    batch_features[col] = encoder.transform(batch_features[col])

            # Align columns
            batch_features = batch_features.reindex(columns=feature_names, fill_value=0)

            # Predict
            batch_predictions = model.predict(batch_features)
            batch_probs = model.predict_proba(batch_features)[:,1]
            batch_data['Churn_Prediction'] = np.where(batch_predictions==1, 'Yes', 'No')
            batch_data['Churn_Probability'] = batch_probs

            st.subheader("Prediction Results")
            st.dataframe(batch_data, height=300)

            st.subheader("Churn Statistics")
            st.bar_chart(batch_data['Churn_Prediction'].value_counts())
