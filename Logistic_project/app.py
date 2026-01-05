import streamlit as st
import pandas as pd
import joblib

import os

@st.cache_resource
def load_model_and_encoder():
    base_dir = os.path.dirname(__file__)

    model = joblib.load(os.path.join(base_dir, "titanic_model.pkl"))
    le_sex = joblib.load(os.path.join(base_dir, "sex_encoder.pkl"))
    feature_names = joblib.load(os.path.join(base_dir, "feature_names.pkl"))

    return model, le_sex, feature_names

def main():
    st.title("Titanic Survival Prediction (Joblib)")

    model, le_sex, feature_names = load_model_and_encoder()

    st.sidebar.header("Passenger inputs")

    pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
    sex_str = st.sidebar.selectbox("Sex", ["male", "female"])
    age = st.sidebar.slider("Age", 0, 80, 25)
    sibsp = st.sidebar.number_input("Siblings/Spouses aboard (SibSp)", 0, 10, 0)
    parch = st.sidebar.number_input("Parents/Children aboard (Parch)", 0, 10, 0)
    fare = st.sidebar.number_input("Fare", 0.0, 600.0, 7.25)

    if st.button("Predict"):
        # Create raw input
        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [sex_str],
            "Pclass": [pclass],
            "Fare": [fare],
            "SibSp": [sibsp],
            "Parch": [parch],
        })

        # Apply saved encoder to Sex
        input_df["Sex"] = le_sex.transform(input_df["Sex"])

        # Ensure column order is same as training
        input_df = input_df[feature_names]

        prob_survive = model.predict_proba(input_df)[0][1]
        pred_class = model.predict(input_df)[0]

        st.subheader("Result")
        st.write(f"Survival probability: **{prob_survive:.3f}**")
        st.write(f"Predicted class (0 = did not survive, 1 = survived): **{pred_class}**")

if __name__ == "__main__":
    main()
