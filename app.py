import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="CCPP Power Prediction",
    page_icon="⚡",
    layout="wide"
)

# Load trained model
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "ccpp_model.joblib")

model = load_model()

# Load model metrics
with open(BASE_DIR / "models" / "model_metrics.json") as f:
    metrics = json.load(f)


# Application title
st.title("⚡ Combined Cycle Power Plant")
st.subheader("Electrical Power Output Prediction")

st.write(
    "Predict electrical power output (PE) using "
    "four environmental and operational parameters."
)

st.divider()

# Model performance
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{metrics['MAE']:.2f} MW")
col2.metric("RMSE", f"{metrics['RMSE']:.2f} MW")
col3.metric("R² Score", f"{metrics['R2 Score']:.4f}")

st.divider()

# Input section
st.subheader("Enter Input Parameters")

col1, col2 = st.columns(2)

with col1:
    AT = st.number_input(
        "Ambient Temperature (AT)",
        min_value=2.0,
        max_value=37.0,
        value=20.0
    )

    V = st.number_input(
        "Exhaust Vacuum (V)",
        min_value=25.0,
        max_value=82.0,
        value=50.0
    )

with col2:
    AP = st.number_input(
        "Ambient Pressure (AP)",
        min_value=992.0,
        max_value=1035.0,
        value=1013.0
    )

    RH = st.number_input(
        "Relative Humidity (RH)",
        min_value=25.0,
        max_value=100.0,
        value=75.0
    )

# Prediction
if st.button("Predict Power Output", type="primary"):

    input_data = pd.DataFrame({
        'AT': [AT],
        'V': [V],
        'AP': [AP],
        'RH': [RH]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction Generated Successfully!")

    st.metric(
        label="Predicted Electrical Power Output",
        value=f"{prediction:.2f} MW"
    )

    st.info(
        "Prediction generated using the Tuned Random Forest model."
    )

st.divider()

st.caption(
    "Machine Learning Project | Combined Cycle Power Plant "
    "| Random Forest Regression"
)