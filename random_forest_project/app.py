import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- Load model & encoder ----------
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("breast_cancer_rf_with_encoder.pkl")
    model = artifacts["model"]
    le = artifacts["label_encoder"]
    return model, le

rf, le = load_artifacts()

# ---------- Load metadata (feature names) ----------
# Use the same CSV to get column order
df = pd.read_csv("breast-cancer.csv")
if 'id' in df.columns:
    df = df.drop(columns=['id'])

target_col = 'diagnosis'
feature_cols = [c for c in df.columns if c != target_col]

st.title("Breast Cancer Prediction App")
st.write("Random Forest classifier trained on breast cancer features.")  # [file:1]

st.sidebar.header("Input Features")

# ---------- Collect user input ----------
user_data = {}

for col in feature_cols:
    # Use min/max/mean from dataset to make sensible defaults
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    col_mean = float(df[col].mean())

    user_data[col] = st.sidebar.number_input(
        label=col,
        min_value=col_min,
        max_value=col_max,
        value=col_mean,
        format="%.4f"
    )

# Convert to dataframe with single row
input_df = pd.DataFrame([user_data], columns=feature_cols)

st.subheader("Input summary")
st.dataframe(input_df)

# ---------- Prediction ----------
if st.button("Predict"):
    # Model outputs encoded labels (0/1), convert back to M/B
    pred_encoded = rf.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    proba = rf.predict_proba(input_df)[0]
    class_names = le.inverse_transform(np.array([0, 1]))

    st.subheader("Prediction")
    diagnosis_text = "Malignant" if pred_label == "M" else "Benign"
    st.write(f"**Predicted diagnosis:** {diagnosis_text} (label: {pred_label})")

    st.subheader("Probability")
    st.write(f"{class_names[0]}: {proba[0]:.4f}")
    st.write(f"{class_names[1]}: {proba[1]:.4f}")
