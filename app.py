import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm

st.set_page_config(page_title="Maize Yield Predictor - Rwanda", layout="centered")

st.title("🌽 Maize Yield Predictor for Rwanda")
st.markdown("Hybrid Model: ARIMAX(1,2,1) + SVR")
st.markdown("---")

# Load pre-trained models using pickle
@st.cache_resource
def load_models():
    with open('arimax_model.pkl', 'rb') as f:
        arimax_model = pickle.load(f)
    with open('svr_model.pkl', 'rb') as f:
        svr_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return arimax_model, svr_model, scaler

try:
    arimax_model, svr_model, scaler = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Models not found. Please upload the model files first. Error: {e}")

if models_loaded:
    st.subheader("Enter Climate Data:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=40.0, value=22.5, step=0.1)
    
    with col2:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=150.0, step=5.0)
    
    if st.button("🌽 Predict Maize Yield", type="primary"):
        # Prepare input
        input_data = np.array([[temperature, rainfall]])
        input_scaled = scaler.transform(input_data)
        input_with_const = sm.add_constant(input_scaled)
        
        # ARIMAX prediction
        arimax_pred = arimax_model.forecast(steps=1, exog=input_with_const)[0]
        
        # SVR prediction
        svr_pred = svr_model.predict(input_scaled)[0]
        
        # Hybrid prediction
        hybrid_pred = arimax_pred + svr_pred
        
        # Display
        st.markdown("---")
        st.subheader("📊 Prediction Result:")
        
        col1, col2, col3 = st.columns(3)
        with col2:
            st.metric("🌽 Predicted Yield", f"{hybrid_pred:,.0f} kg/ha")
        
        st.info(f"📌 ARIMAX: {arimax_pred:,.0f} + SVR: {svr_pred:,.0f} = {hybrid_pred:,.0f} kg/ha")

st.markdown("---")
st.caption("HYBRID MODEL: ARIMAX(1,2,1) + SVR | Trained on Rwanda data")
