# app_final_consistent.py - Uses batch method for all predictions
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Maize Yield Predictor - Rwanda", layout="wide")

st.title("🌽 Maize Yield Predictor for Rwanda")
st.markdown("**HYBRID MODEL: ARIMAX(1,2,1) + SVR**")
st.markdown("---")

@st.cache_resource
def train_model():
    # Load cleaned data
    data = pd.read_csv("rwanda_climate_maize_clean.csv")
    return data

@st.cache_resource
def train_and_predict(future_years_data):
    """Train model and predict including future years"""
    
    # Load base data
    data = pd.read_csv("rwanda_climate_maize_clean.csv")
    
    # Append future years
    data_extended = pd.concat([data, future_years_data], ignore_index=True)
    
    # Split
    split = int(len(data) * 0.85)
    train = data_extended.iloc[:split]
    test = data_extended.iloc[split:]
    
    y_train = train["maize_yield"].dropna()
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train[["temperature_C", "rainfall"]])
    constant_col = np.ones((X_train_scaled.shape[0], 1))
    X_train = np.hstack([constant_col, X_train_scaled])
    
    # Train ARIMAX
    model = ARIMA(y_train, exog=X_train, order=(1, 2, 1))
    fit = model.fit()
    
    # Train SVR
    residuals = y_train - fit.fittedvalues
    residuals_smoothed = residuals.rolling(3, min_periods=1).mean().bfill()
    svr = SVR(kernel="rbf", C=200, epsilon=0.1, gamma="scale")
    svr.fit(X_train_scaled, residuals_smoothed)
    
    # Predict all test years
    X_test_scaled = scaler.transform(test[["temperature_C", "rainfall"]])
    constant_test = np.ones((X_test_scaled.shape[0], 1))
    X_test = np.hstack([constant_test, X_test_scaled])
    arimax_test_pred = fit.forecast(steps=len(test), exog=X_test)
    
    # Store predictions
    predictions = {}
    for i in range(len(test)):
        year = int(test.iloc[i]['year'])
        predictions[year] = {
            'arimax': arimax_test_pred.iloc[i],
            'temp': test.iloc[i]['temperature_C'],
            'rain': test.iloc[i]['rainfall']
        }
    
    return fit, svr, scaler, predictions, test

# Load base data
data = pd.read_csv("rwanda_climate_maize_clean.csv")

# UI - Historical vs Future
st.subheader("Select Prediction Mode")

mode = st.radio("Choose mode:", ["Historical Year (1961-2025)", "Future Year (2026+)"], horizontal=True)

if mode == "Historical Year (1961-2025)":
    # For historical, use pre-computed test set
    # Create empty future years data
    future_data = pd.DataFrame(columns=['year', 'temperature_C', 'rainfall', 'maize_yield'])
    fit, svr, scaler, predictions, test = train_and_predict(future_data)
    
    years = sorted(data['year'].unique())
    selected_year = st.selectbox("Select Year", years, index=len(years)-1)
    
    if selected_year in predictions:
        pred = predictions[selected_year]
        temp = pred['temp']
        rain = pred['rain']
        
        scaled = scaler.transform(np.array([[temp, rain]]))
        svr_pred = svr.predict(scaled)[0]
        hybrid = pred['arimax'] + svr_pred
        
        actual = data[data['year'] == selected_year]['maize_yield'].values[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temperature", f"{temp:.2f}°C")
        with col2:
            st.metric("☔ Rainfall", f"{rain:.1f} mm")
        with col3:
            st.metric(" Actual Yield", f"{actual:.1f} kg/ha")
        
        if st.button("Predict", type="primary"):
            st.markdown("---")
            st.subheader("Prediction Results")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("🌽 HYBRID", f"{hybrid:.0f} kg/ha")
            with c2:
                st.metric(" ARIMAX", f"{pred['arimax']:.0f} kg/ha")
            with c3:
                st.metric(" SVR", f"{svr_pred:+.0f} kg/ha")
            
            error = hybrid - actual
            error_pct = (error / actual) * 100
            st.metric("Difference", f"{error:+.0f} kg/ha", delta=f"{error_pct:+.1f}%", delta_color="inverse")

else:
    # Future years
    st.subheader("Predict Future Year")
    
    col1, col2 = st.columns(2)
    with col1:
        future_year = st.number_input("Year", min_value=2026, max_value=2050, value=2026, step=1)
    with col2:
        st.write("")
    
    col1, col2 = st.columns(2)
    with col1:
        temp_input = st.number_input("🌡️ Expected Temperature (°C)", min_value=15.0, max_value=30.0, value=20.6, step=0.1, format="%.2f")
    with col2:
        rain_input = st.number_input("☔ Expected Rainfall (mm)", min_value=800.0, max_value=1600.0, value=1280.0, step=10.0, format="%.1f")
    
    if st.button("Predict Future Year", type="primary"):
        # Create future year data and retrain
        future_data = pd.DataFrame([{
            'year': future_year,
            'temperature_C': temp_input,
            'rainfall': rain_input,
            'maize_yield': np.nan
        }])
        
        with st.spinner("Calculating prediction..."):
            fit, svr, scaler, predictions, test = train_and_predict(future_data)
        
        pred = predictions[future_year]
        scaled = scaler.transform(np.array([[temp_input, rain_input]]))
        svr_pred = svr.predict(scaled)[0]
        hybrid = pred['arimax'] + svr_pred
        
        st.markdown("---")
        st.subheader(f"Prediction for {int(future_year)}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🌽 HYBRID", f"{hybrid:.0f} kg/ha")
        with c2:
            st.metric(" ARIMAX", f"{pred['arimax']:.0f} kg/ha")
        with c3:
            st.metric(" SVR", f"{svr_pred:+.0f} kg/ha")
        
        # Compare with 2025
        hist_mean = data['maize_yield'].mean()
        if hybrid > hist_mean:
            st.success(f"Above historical average ({hist_mean:.0f} kg/ha)")
        else:
            st.info(f"Below historical average ({hist_mean:.0f} kg/ha)")

st.caption("**Thesis:** Hybrid ARIMAX(1,2,1) + SVR | Trained 1982-2015 | Tested 2016-2025")
