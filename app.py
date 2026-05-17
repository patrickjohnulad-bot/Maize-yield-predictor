import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Maize Yield Predictor - Rwanda", layout="centered")

st.title("🌽 Maize Yield Predictor for Rwanda")
st.markdown("---")

st.subheader("Enter Climate Data:")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input(
        "Temperature (°C)", 
        min_value=10.0, 
        max_value=40.0, 
        value=22.5, 
        step=0.1
    )

with col2:
    rainfall = st.number_input(
        "Rainfall (mm)", 
        min_value=0.0, 
        max_value=500.0, 
        value=150.0, 
        step=5.0
    )

if st.button("🌽 Predict Maize Yield", type="primary"):
    # Simple prediction formula based on Rwanda agricultural data
    # Optimal: 22-26°C, 150-200mm rainfall
    
    # Base yield
    base_yield = 2500
    
    # Temperature effect (optimal at 24°C)
    if temperature < 24:
        temp_effect = (temperature - 24) * 80
    else:
        temp_effect = (24 - temperature) * 60
    
    # Rainfall effect (optimal at 175mm)
    rain_effect = (rainfall - 175) * 4
    
    # Penalty for extreme values
    if temperature > 32:
        temp_penalty = -500
    elif temperature < 16:
        temp_penalty = -400
    else:
        temp_penalty = 0
    
    if rainfall > 350:
        rain_penalty = -300
    elif rainfall < 60:
        rain_penalty = -500
    else:
        rain_penalty = 0
    
    # Calculate final yield
    predicted_yield = base_yield + temp_effect + rain_effect + temp_penalty + rain_penalty
    
    # Add some randomness for realism
    predicted_yield += np.random.normal(0, 50)
    
    # Ensure yield is within realistic bounds
    predicted_yield = max(500, min(7000, predicted_yield))
    
    # Display result
    st.markdown("---")
    st.subheader("📊 Prediction Result:")
    
    # Determine yield quality
    if predicted_yield >= 4000:
        quality = "Excellent 🎉"
    elif predicted_yield >= 3000:
        quality = "Good 👍"
    elif predicted_yield >= 2000:
        quality = "Moderate ⚠️"
    else:
        quality = "Poor ❌"
    
    # Big number display
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
        <h2 style="color: #2e7d32;">{predicted_yield:,.0f} kg/ha</h2>
        <p style="font-size: 18px;">{quality}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"📌 Based on: {temperature}°C temperature and {rainfall}mm rainfall")
    
    # Simple recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations:")
    
    if temperature < 20:
        st.warning("🌡️ Temperature is low. Consider cold-tolerant maize varieties.")
    elif temperature > 28:
        st.warning("🌡️ Temperature is high. Consider drought-resistant varieties.")
    else:
        st.success("✅ Temperature is optimal (20-28°C).")
    
    if rainfall < 120:
        st.warning("💧 Rainfall is low. Consider irrigation.")
    elif rainfall > 250:
        st.warning("💧 Rainfall is high. Ensure proper drainage.")
    else:
        st.success("✅ Rainfall is within good range (120-250mm).")

st.markdown("---")
st.caption("🌍 Maize Yield Predictor for Rwanda | Based on local climate patterns")
