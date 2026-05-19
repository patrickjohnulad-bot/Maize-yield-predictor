# app.py - Using sklearn LinearRegression instead of statsmodels
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Maize Yield Predictor - Rwanda", layout="wide")

st.title("🌽 Maize Yield Predictor for Rwanda")
st.markdown("**HYBRID MODEL: ARIMAX(1,2,1) + SVR**")
st.markdown("---")

@st.cache_data
def load_data():
    data = pd.read_csv("rwanda_climate_maize_clean.csv")
    return data

@st.cache_resource
def train_base_model():
    data = load_data()
    
    # Train: 1982-2015 (34 years)
    train_data = data[(data['year'] >= 1982) & (data['year'] <= 2015)].copy()
    test_data = data[data['year'] >= 2016].copy()
    
    y_train = train_data["maize_yield"].values
    
    # Create lag features for AR terms
    y_lag1 = np.roll(y_train, 1)
    y_lag1[0] = y_train[0]
    y_lag2 = np.roll(y_train, 2)
    y_lag2[0:2] = y_train[0:2]
    
    # Scale climate features
    scaler = RobustScaler()
    X_climate_scaled = scaler.fit_transform(train_data[["temperature_C", "rainfall"]])
    
    # Create ARIMAX features (lags + climate)
    X_train = np.column_stack([y_lag1, y_lag2, X_climate_scaled])
    
    # Train Linear Model (AR part)
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    # Get predictions and residuals
    arimax_pred_train = linear_model.predict(X_train)
    residuals = y_train - arimax_pred_train
    
    # Train SVR on residuals
    residuals_smoothed = pd.Series(residuals).rolling(3, min_periods=1).mean().bfill().values
    svr = SVR(kernel="rbf", C=200, epsilon=0.1, gamma="scale")
    svr.fit(X_train, residuals_smoothed)
    
    # Predict test years
    test_years = test_data['year'].values
    X_test_climate = scaler.transform(test_data[["temperature_C", "rainfall"]])
    
    # Build test features
    test_predictions = {}
    last_y1 = y_train[-1]
    last_y2 = y_train[-2]
    
    for i, year in enumerate(test_years):
        # Create lag features for test
        if i == 0:
            y_lag1_test = last_y1
            y_lag2_test = last_y2
        else:
            y_lag1_test = hybrid_pred_prev
            y_lag2_test = last_y1
        
        X_test_row = np.column_stack([[y_lag1_test, y_lag2_test, X_test_climate[i]]])
        
        arimax_pred = linear_model.predict(X_test_row)[0]
        svr_pred = svr.predict(X_test_row)[0]
        hybrid_pred = arimax_pred + svr_pred
        
        test_predictions[year] = {
            'arimax': float(arimax_pred),
            'svr': float(svr_pred),
            'hybrid': float(hybrid_pred),
            'temp': float(test_data.iloc[i]['temperature_C']),
            'rain': float(test_data.iloc[i]['rainfall'])
        }
        
        hybrid_pred_prev = hybrid_pred
        last_y1 = hybrid_pred
        last_y2 = y_lag1_test
    
    return linear_model, svr, scaler, test_predictions, train_data

data = load_data()
linear_model, svr, scaler, test_predictions, train_data = train_base_model()

hist_mean = train_data['maize_yield'].mean()

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
    st.subheader("Predict Future Year")
    
    future_year = st.number_input("Year", min_value=2026, max_value=2050, value=2026, step=1)
    
    col1, col2 = st.columns(2)
    with col1:
        temp_input = st.number_input("🌡️ Expected Temperature (°C)", min_value=15.0, max_value=30.0, value=20.6, step=0.1, format="%.2f")
    with col2:
        rain_input = st.number_input("☔ Expected Rainfall (mm)", min_value=800.0, max_value=1600.0, value=1280.0, step=10.0, format="%.1f")
    
    if st.button("Predict Future Year", type="primary"):
        with st.spinner("Calculating prediction..."):
            # Use last two predictions from test set
            last_year = max(test_predictions.keys())
            last_pred = test_predictions[last_year]
            prev_year = last_year - 1
            
            if prev_year in test_predictions:
                prev_pred = test_predictions[prev_year]
                y_lag1 = last_pred['hybrid']
                y_lag2 = prev_pred['hybrid']
            else:
                y_lag1 = last_pred['hybrid']
                y_lag2 = y_lag1 - 50
            
            X_climate = np.array([[temp_input, rain_input]])
            X_climate_scaled = scaler.transform(X_climate)
            
            X_future = np.column_stack([[y_lag1, y_lag2, X_climate_scaled[0]]])
            
            arimax_future = linear_model.predict(X_future)[0]
            svr_future = svr.predict(X_future)[0]
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
        
        if hybrid_future > hist_mean:
            st.success(f"✅ Above historical average ({hist_mean:.0f} kg/ha)")
        else:
            st.info(f"📉 Below historical average ({hist_mean:.0f} kg/ha)")

st.caption("**Thesis:** Hybrid ARIMAX(1,2,1) + SVR | Trained 1982-2015 | Tested 2016-2025")
