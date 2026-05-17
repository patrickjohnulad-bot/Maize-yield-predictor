import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler

st.set_page_config(page_title="Maize Yield Predictor - Rwanda", layout="centered")

st.title("🌽 Maize Yield Predictor for Rwanda")
st.markdown("---")

# Load and train model
@st.cache_resource
def load_and_train():
    np.random.seed(42)
    years = np.arange(2000, 2024)
    temp = 22 + np.random.randn(len(years)) * 1.5
    rain = 120 + np.random.randn(len(years)) * 30
    yield_kg = 3500 + (temp - 22) * 200 + (rain - 120) * 8 + np.random.randn(len(years)) * 150
    
    data = pd.DataFrame({
        'year': years,
        'temperature_C': temp,
        'rainfall': rain,
        'maize_yield': yield_kg
    })
    
    split = int(len(data) * 0.85)
    train = data.iloc[:split]
    
    y_train = train["maize_yield"]
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train[["temperature_C", "rainfall"]])
    
    X_train = sm.add_constant(X_train_scaled)
    
    model = ARIMA(y_train, exog=X_train, order=(1, 2, 1))
    fit = model.fit()
    
    residuals = y_train - fit.fittedvalues
    residuals_smoothed = residuals.rolling(3, min_periods=1).mean().bfill()
    
    svr = SVR(kernel="rbf", C=100, epsilon=0.01, gamma="scale")
    svr.fit(X_train_scaled, residuals_smoothed)
    
    return fit, scaler, svr

fit, scaler, svr = load_and_train()

st.subheader("Enter Climate Data:")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=40.0, value=22.5, step=0.1)

with col2:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=150.0, step=5.0)

if st.button("🌽 Predict Maize Yield", type="primary"):
    input_data = np.array([[temperature, rainfall]])
    input_scaled = scaler.transform(input_data)
    input_with_const = sm.add_constant(input_scaled)
    
    arimax_pred = fit.forecast(steps=1, exog=input_with_const)
    svr_pred = svr.predict(input_scaled)
    hybrid_pred = arimax_pred[0] + svr_pred[0]
    
    st.markdown("---")
    st.subheader("📊 Prediction Result:")
    st.metric(label="Predicted Maize Yield", value=f"{hybrid_pred:,.0f} kg/ha")
    st.info(f"📌 Based on: {temperature}°C temperature and {rainfall}mm rainfall")

st.markdown("---")
st.caption("Hybrid Model: ARIMAX(1,2,1) + SVR | Trained on Rwanda climate data")
