"""
Medical Insurance Cost Prediction - Streamlit App
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="💊", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("outputs/best_model.joblib")
    with open("outputs/best_model_info.json") as f:
        info = json.load(f)
    return model, info

model, info = load_model()

st.title("💊 Medical Insurance Cost Predictor")
st.caption(f"Powered by **{info['best_model']}**  |  Test R² = {info['metrics']['R2']}  |  RMSE = ₹{info['metrics']['RMSE']:,.0f}")

st.write("Enter the details below to estimate annual medical insurance charges.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 64, 30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=55.0, value=25.0, step=0.1)
    children = st.slider("Number of Children", 0, 5, 0)

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict Cost", type="primary"):
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"### Estimated Annual Insurance Charges: **₹{prediction:,.2f}**")

    if smoker == "yes":
        st.warning("Smoking status significantly increases predicted charges.")
    if bmi >= 30:
        st.info("BMI is in the obese range, which tends to raise predicted charges.")

st.divider()
st.caption("Medical Insurance Cost Prediction")
