import streamlit as st
import numpy as np
import joblib
import os

# ---------------------------------
# Load trained objects
# ---------------------------------
@st.cache_resource
def load_objects():
    base_dir = os.path.dirname(__file__)

    model = joblib.load(os.path.join(base_dir, "decision_tree_model.pkl"))
    imputer = joblib.load(os.path.join(base_dir, "imputer.pkl"))
    label_encoders = joblib.load(os.path.join(base_dir, "label_encoders.pkl"))

    return model, imputer, label_encoders

model, imputer, label_encoders = load_objects()

# ---------------------------------
# Feature order (EXACT training order)
# ---------------------------------
FEATURE_ORDER = [
    "name",
    "city",
    "income",
    "credit_score",
    "loan_amount",
    "years_employed",
    "points"
]

# ---------------------------------
# UI
# ---------------------------------
st.title("🏦 Loan Approval Prediction")
st.write("Decision Tree Model")

st.divider()
st.subheader("Enter Applicant Details")

# ---------------------------------
# Inputs
# ---------------------------------
input_values = []

for col in FEATURE_ORDER:
    if col in label_encoders:   # categorical columns (name, city)
        value = st.selectbox(
            col.capitalize(),
            label_encoders[col].classes_.tolist()
        )
        value = label_encoders[col].transform([value])[0]
    else:                       # numerical columns
        value = st.number_input(col.capitalize())
    input_values.append(value)

# Convert to array
input_array = np.array([input_values])

# Apply same imputer
input_array = imputer.transform(input_array)

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array).max()

    if prediction == 1:
        st.success(f"✅ Loan Approved (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Confidence: {probability:.2f})")
