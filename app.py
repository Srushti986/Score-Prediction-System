# Streamlit File
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(
    page_title="Student Score Predictor",
    layout="centered"
)

st.title("Score Prediction System Based on Hours")
st.write("Predict the student's score based on the number of study hours.")

st.divider()

hours = st.number_input(
    "Enter study hours:",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Score", type="primary"):
    input_data = pd.DataFrame({"Hours": [hours]})

    # scaled_input = scaler.transform(input_data)
    prediction = model.predict(input_data)[0]

    predicted_score = float(prediction)
    predicted_score = max(0.0, min(100.0, predicted_score))

    st.success(f"Predicted Score Based on Hours: {predicted_score:.2f}")

    if predicted_score >= 75:
        st.info("High predicted score")
        st.balloons()
    elif predicted_score >= 50:
        st.info("Moderate predicted score")
    else:
        st.info("More study may improve the predicted score")

st.divider()

with st.expander("Model Information"):
    st.write("**Model:** Linear Regression")
    st.write("**Input feature:** Hours")