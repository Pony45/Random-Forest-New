import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="XGBoost M&V", layout="wide")
st.title("🚀 XGBoost M&V Dashboard")
st.markdown("*AI-based Measurement & Verification using XGBoost*")

# Load model with pickle
@st.cache_resource
def load_model():
    model_path = 'models/xgboost_model.pkl'
    features_path = 'models/xgboost_features.txt'
    
    if not os.path.exists(model_path):
        st.error("❌ Model not found!")
        return None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

st.success("✅ XGBoost model loaded!")

# Sidebar
with st.sidebar:
    st.header("📋 Building Parameters")
    
    temp = st.slider("🌡️ Temperature (°C)", 22, 35, 28)
    humidity = st.slider("💧 Humidity (%)", 60, 95, 80)
    hour = st.slider("⏰ Hour", 0, 23, 14)
    dayofweek = st.selectbox("📅 Day", list(range(7)), format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
    month = st.selectbox("📆 Month", list(range(1,13)), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
    floor_area = st.number_input("🏠 Floor Area (m²)", 50, 300, 120)
    occupants = st.number_input("👥 Occupants", 1, 8, 4)
    retrofit = st.selectbox("🔧 Retrofit", [0,1], format_func=lambda x: "✅ Yes" if x else "❌ No")

# Feature engineering
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
is_weekend = 1 if dayofweek >= 5 else 0
temp_humidity = temp * humidity / 100
occ_per_area = occupants / floor_area

X = pd.DataFrame([[
    temp, humidity, hour, dayofweek, month, floor_area, occupants, retrofit,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend, temp_humidity, occ_per_area
]], columns=FEATURES)

# Predict
if st.button("🔮 Predict", type="primary"):
    pred = model.predict(X)[0] / 15
    st.metric("⚡ Predicted Energy", f"{pred:.2f} kWh")
    
    if retrofit == 1:
        X_base = X.copy()
        X_base['retrofit'] = 0
        base = model.predict(X_base)[0] / 15
        savings = base - pred
        pct = (savings / base) * 100
        st.success(f"💰 Savings: {savings:.2f} kWh ({pct:.1f}%)")
        st.info(f"Monthly Savings: RM {savings * 24 * 30 * 0.52:.2f}")
    else:
        X_retro = X.copy()
        X_retro['retrofit'] = 1
        retro_pred = model.predict(X_retro)[0] / 15
        potential = pred - retro_pred
        st.info(f"💡 If retrofitted: Save ~{potential:.2f} kWh")

st.markdown("---")
st.caption("🚀 XGBoost M&V System | Thesis Project | Scaled for Malaysian Residential Buildings")
