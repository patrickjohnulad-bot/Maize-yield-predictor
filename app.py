import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler

st.set_page_config(page_title="Maize Yield Predictor - Rwanda", layout="centered")

st.title("🌽 Maize Yield Predictor for Rwanda")
st.markdown("**HYBRID MODEL: ARIMAX(1,2,1) + SVR**")
st.markdown("---")

@st.cache_resource
def train_model():
    data = pd.read_csv("rwanda_climate_maize_dataset.csv")
    
    cols = ["temperature_C", "rainfall", "maize_yield"]
    data[cols] = data[cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna()
    
    Q1 = data['maize_yield'].quantile(0.25)
    Q3 = data['maize_yield'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['maize_yield'] >= lower_bound) & (data['maize_yield'] <= upper_bound)]
    
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
    
    return fit, svr, scaler

fit, svr, scaler = train_model()

st.subheader("Enter Climate Data:")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", 10.0, 40.0, 22.5, 0.1)

with col2:
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 150.0, 5.0)

if st.button("🌽 Predict Maize Yield", type="primary"):
    input_data = np.array([[temperature, rainfall]])
    input_scaled = scaler.transform(input_data)
    input_with_const = sm.add_constant(input_scaled)
    
    arimax_pred = fit.forecast(steps=1, exog=input_with_const)[0]
    svr_pred = svr.predict(input_scaled)[0]
    hybrid_pred = arimax_pred + svr_pred
    
    st.markdown("---")
    st.subheader("📊 Prediction Result:")
    st.markdown(f"<h1 style='text-align: center; color: #2e7d32;'>{hybrid_pred:,.0f} kg/ha</h1>", unsafe_allow_html=True)
    st.caption(f"ARIMAX(1,2,1): {arimax_pred:,.0f} kg/ha + SVR correction: {svr_pred:,.0f} kg/ha")

st.markdown("---")
st.caption("Thesis: Hybrid ARIMAX-SVR Model for Maize Yield Prediction in Rwanda")
