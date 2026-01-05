import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

# -------------------------------
# Load model & scaler (SAFE)
# -------------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    kmeans = joblib.load(os.path.join(base_dir, "kmeans_mall_customers.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "scaler_mall_customers.pkl"))
    return kmeans, scaler

kmeans, scaler = load_artifacts()

# -------------------------------
# UI
# -------------------------------
st.title("🛍️ Mall Customer Segmentation")
st.write("KMeans clustering model")

st.divider()
st.subheader("Enter Customer Details")

# -------------------------------
# Inputs (MATCH TRAINING DATA)
# -------------------------------
# Typical Mall Customers dataset:
# [Age, Annual Income (k$), Spending Score (1-100)]

annual_income = st.number_input("Annual Income (k$)", min_value=5, max_value=200, value=60)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Customer Segment"):
    input_data = np.array([[annual_income, spending_score]])

    # Apply same scaling used during training
    input_scaled = scaler.transform(input_data)

    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"✅ Customer belongs to **Cluster {cluster}**")
