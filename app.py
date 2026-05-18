# app_final_consistent.py - Simplified working version
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Maize Yield Predictor - Rwanda", layout="wide")

st.title("🌽 Maize Yield Predictor for Rwanda")
st.markdown("**HYBRID MODEL: ARIMAX(1,2,1) + SVR**")
st.markdown("---")

# Load data once
@st.cache_data
def load_data():
    data = pd.read_csv("rwanda_climate_maize_clean.csv")
    return data

# Train model on historical data only (no future years in training)
@st.cache_resource
def train_base_model():
    data = load_data()
    
    # Split at 2015 (85% of 66 years = 56 years)
    train_data = data[data['year'] <= 2015].copy()
    test_data = data[data['year'] >= 2016].copy()
    
    y_train = train_data["maize_yield"]
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_data[["temperature_C", "rainfall"]])
    constant_col = np.ones((X_train_scaled.shape[0], 1))
    X_train = np.hstack([constant_col, X_train_scaled])
    
    # Train ARIMAX
    model = ARIMA(y_train, exog=X_train, order=(1, 2, 1))
    fit = model.fit()
    
    # Train SVR on residuals
    residuals = y_train - fit.fittedvalues
    residuals_smoothed = residuals.rolling(3, min_periods=1).mean().bfill()
    svr = SVR(kernel="rbf", C=200, epsilon=0.1, gamma="scale")
    svr.fit(X_train_scaled, residuals_smoothed)
    
    # Get predictions for historical test years (2016-2025)
    X_test_scaled = scaler.transform(test_data[["temperature_C", "rainfall"]])
    constant_test = np.ones((X_test_scaled.shape[0], 1))
    X_test = np.hstack([constant_test, X_test_scaled])
    arimax_pred = fit.forecast(steps=len(test_data), exog=X_test)
    svr_pred = svr.predict(X_test_scaled)
    hybrid_pred = arimax_pred + svr_pred
    
    # Store test predictions by year
    test_predictions = {}
    for i, year in enumerate(test_data['year'].values):
        test_predictions[year] = {
            'arimax': arimax_pred.iloc[i],
            'svr': svr_pred[i],
            'hybrid': hybrid_pred[i],
            'temp': test_data.iloc[i]['temperature_C'],
            'rain': test_data.iloc[i]['rainfall']
        }
    
    return fit, svr, scaler, train_data, test_predictions

# Load base data
data = load_data()

# Train base model once
fit, svr, scaler, train_data, test_predictions = train_base_model()

# UI - Historical vs Future
st.subheader("Select Prediction Mode")

mode = st.radio("Choose mode:", ["Historical Year (2016-2025)", "Future Year (2026+)"], horizontal=True)

if mode == "Historical Year (2016-2025)":
    st.subheader("Select Year")
    
    years = sorted(test_predictions.keys())
    selected_year = st.selectbox("Year", years, index=len(years)-1)
    
    pred = test_predictions[selected_year]
    actual = data[data['year'] == selected_year]['maize_yield'].values[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌡️ Temperature", f"{pred['temp']:.2f}°C")
    with col2:
        st.metric("☔ Rainfall", f"{pred['rain']:.1f} mm")
    with col3:
        st.metric("📊 Actual Yield", f"{actual:.1f} kg/ha")
    
    if st.button("Predict", type="primary"):
        st.markdown("---")
        st.subheader("Prediction Results")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🌽 HYBRID", f"{pred['hybrid']:.0f} kg/ha")
        with c2:
            st.metric("📈 ARIMAX", f"{pred['arimax']:.0f} kg/ha")
        with c3:
            st.metric("⚙️ SVR", f"{pred['svr']:+.0f} kg/ha")
        
        error = pred['hybrid'] - actual
        error_pct = (error / actual) * 100
        st.metric("Difference", f"{error:+.0f} kg/ha", delta=f"{error_pct:+.1f}%", delta_color="inverse")

else:
    # Future years
    st.subheader("Predict Future Year")
    
    col1, col2 = st.columns(2)
    with col1:
        future_year = st.number_input("Year", min_value=2026, max_value=2050, value=2026, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        temp_input = st.number_input("🌡️ Expected Temperature (°C)", min_value=15.0, max_value=30.0, value=20.6, step=0.1, format="%.2f")
    with col2:
        rain_input = st.number_input("☔ Expected Rainfall (mm)", min_value=800.0, max_value=1600.0, value=1280.0, step=10.0, format="%.1f")
    
    if st.button("Predict Future Year", type="primary"):
        with st.spinner("Calculating prediction..."):
            # Scale the input
            scaled_input = scaler.transform(np.array([[temp_input, rain_input]]))
            constant_input = np.ones((1, 1))
            X_input = np.hstack([constant_input, scaled_input])
            
            # Get ARIMAX prediction
            arimax_future = fit.forecast(steps=1, exog=X_input)[0]
            
            # Get SVR prediction
            svr_future = svr.predict(scaled_input)[0]
            
            # Hybrid prediction
            hybrid_future = arimax_future + svr_future
        
        st.markdown("---")
        st.subheader(f"Prediction for {int(future_year)}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🌽 HYBRID", f"{hybrid_future:.0f} kg/ha")
        with c2:
            st.metric("📈 ARIMAX", f"{arimax_future:.0f} kg/ha")
        with c3:
            st.metric("⚙️ SVR", f"{svr_future:+.0f} kg/ha")
        
        # Compare with historical average
        hist_mean = data['maize_yield'].mean()
        if hybrid_future > hist_mean:
            st.success(f"✅ Above historical average ({hist_mean:.0f} kg/ha)")
        else:
            st.info(f"📉 Below historical average ({hist_mean:.0f} kg/ha)")

st.caption("**Thesis:** Hybrid ARIMAX(1,2,1) + SVR | Trained 1960-2015 | Tested 2016-2025")
