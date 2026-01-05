# app.py
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Ames House Price Predictor", layout="centered")

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    artifacts = joblib.load(
        os.path.join(base_dir, "ames_rf_regressor.joblib")
    )
    return artifacts["model"], artifacts["feature_columns"]

rf_model, feature_cols = load_model()

st.title("Ames House Price Predictor")

st.markdown("Enter house details below to **predict** the SalePrice.")

# --- Basic numeric inputs (add more as needed) ---
col1, col2 = st.columns(2)
with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=200, max_value=6000, value=1500)
    garage_cars = st.number_input("Garage Cars", min_value=0, max_value=5, value=2)

with col2:
    year_built = st.number_input("Year Built", min_value=1870, max_value=2020, value=1990)
    total_bsmt_sf = st.number_input("Total Basement SF", min_value=0, max_value=4000, value=800)
    full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=4, value=2)

# --- Some categorical inputs (must match your training columns) ---
neighborhood = st.selectbox(
    "Neighborhood",
    [
        "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert",
        "Sawyer", "NridgHt", "NWAmes", "Timber"
    ]
)

house_style = st.selectbox(
    "House Style",
    ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "1.5Unf"]
)

ms_zoning = st.selectbox(
    "MS Zoning",
    ["RL", "RM", "FV", "RH", "C (all)"]
)

# --- Build a single-row raw dataframe ---
raw_input = {
    "Overall Qual": overall_qual,
    "Gr Liv Area": gr_liv_area,
    "Garage Cars": garage_cars,
    "Year Built": year_built,
    "Total Bsmt SF": total_bsmt_sf,
    "Full Bath": full_bath,
    "Neighborhood": neighborhood,
    "House Style": house_style,
    "MS Zoning": ms_zoning,
}

raw_df = pd.DataFrame([raw_input])

# --- Apply SAME preprocessing as during training ---
# If you used: X = pd.get_dummies(df.drop(["SalePrice", "Id", ...], axis=1), drop_first=True)
# then do the same here:
proc_df = pd.get_dummies(raw_df, drop_first=True)

# Align columns with training feature_cols
proc_df = proc_df.reindex(columns=feature_cols, fill_value=0)

if st.button("Predict Sale Price"):
    prediction = rf_model.predict(proc_df)[0]
    st.success(f"Predicted SalePrice: ${prediction:,.0f}")
    st.caption("Prediction from your trained RandomForestRegressor model.")
