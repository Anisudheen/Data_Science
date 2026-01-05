import streamlit as st
import numpy as np
import joblib
import os

# -------------------------------
# Load saved model and preprocessors
# -------------------------------

@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)

    model = joblib.load(os.path.join(base_dir, "naive_bayes_model.pkl"))
    imputer = joblib.load(os.path.join(base_dir, "imputer.pkl"))
    sex_encoder = joblib.load(os.path.join(base_dir, "sex_encoder.pkl"))
    embarked_encoder = joblib.load(os.path.join(base_dir, "embarked_encoder.pkl"))

    return model, imputer, sex_encoder, embarked_encoder

model, imputer, sex_encoder, embarked_encoder = load_artifacts()

# -------------------------------
# App UI
# -------------------------------
st.title("🚢 Titanic Survival Prediction")
st.write("Naive Bayes Classification Model")

# -------------------------------
# User Inputs
# -------------------------------
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=30.0)
embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

# -------------------------------
# Encode Inputs
# -------------------------------
sex_encoded = sex_encoder.transform([sex])[0]
embarked_encoded = embarked_encoder.transform([embarked])[0]

input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Handle missing values
input_data = imputer.transform(input_data)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Passenger Survived (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Passenger Did Not Survive (Probability: {probability:.2f})")
