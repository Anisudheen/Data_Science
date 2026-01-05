import numpy as np
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    model = joblib.load("linear_model.pkl")
    return model

model = load_model()

st.title("Weather Temperature Prediction")
st.write("Predict **daily temperature** based on hours of sunlight.")

# Input: hours of sunlight
hours = st.number_input(
    "Enter hours of sunlight:",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5,
)

if st.button("Predict temperature"):
    hours_array = np.array([[hours]])

    # Get prediction
    temp_pred = model.predict(hours_array)[0]

    st.success(f"Predicted temperature: {temp_pred:.2f}")
