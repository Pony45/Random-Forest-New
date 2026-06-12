import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="M&V Dashboard", layout="wide")
st.title("🏠 AI-based Measurement & Verification (M&V) Dashboard")
st.markdown("*Predict energy savings for Malaysian residential buildings*")

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model_path = 'models/thesis_mv_random_forest.pkl'
    features_path = 'models/thesis_mv_features.txt'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        return None, None
    
    if not os.path.exists(features_path):
        st.error(f"❌ Features file not found at {features_path}")
        return None, None
    
    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    return model, features

model, FEATURES = load_model()

if model is None:
    st.stop()

# ==========================================
# SCALING FACTOR untuk rumah Malaysia
# ==========================================
SCALING_FACTOR = 15

def scale_prediction(prediction):
    return prediction / SCALING_FACTOR

# ==========================================
# UNIT CONVERSION FUNCTION
# ==========================================
def convert_energy_unit(prediction_kwh, target_unit):
    if target_unit == "Per Hour (kWh)":
        return prediction_kwh, "kWh"
    elif target_unit == "Per Day (kWh)":
        return prediction_kwh * 24, "kWh/day"
    elif target_unit == "Per Month (kWh)":
        return prediction_kwh * 24 * 30, "kWh/month"
    elif target_unit == "Per Year (kWh)":
        return prediction_kwh * 24 * 365, "kWh/year"

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📋 Building Parameters")

# Unit selector
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Display Settings")
unit_option = st.sidebar.selectbox(
    "Energy Unit Display",
    ["Per Hour (kWh)", "Per Day (kWh)", "Per Month (kWh)", "Per Year (kWh)"]
)

st.sidebar.markdown("---")

# Inputs (Malaysia range)
temp = st.sidebar.slider("🌡️ Temperature (°C)", 22, 35, 28)
humidity = st.sidebar.slider("💧 Humidity (%)", 60, 95, 80)
hour = st.sidebar.slider("⏰ Hour of Day", 0, 23, 14)
dayofweek = st.sidebar.selectbox("📅 Day of Week", [0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month = st.sidebar.selectbox("📆 Month", list(range(1,13)), format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
floor_area = st.sidebar.number_input("🏠 Floor Area (m²)", 50, 300, 120)
occupants = st.sidebar.number_input("👥 Occupants", 1, 8, 4)
retrofit = st.sidebar.selectbox("🔧 Retrofit Status", [0,1], format_func=lambda x: "✅ Yes (Retrofitted)" if x else "❌ No (Baseline)")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
is_weekend = 1 if dayofweek >= 5 else 0
temp_humidity = temp * humidity / 100
occ_per_area = occupants / floor_area

features_df = pd.DataFrame([[
    temp, humidity, hour, dayofweek, month, floor_area, occupants, retrofit,
    hour_sin, hour_cos, month_sin, month_cos, is_weekend, temp_humidity, occ_per_area
]], columns=FEATURES)

# ==========================================
# MAIN CONTENT
# ==========================================
st.info("📌 **Note:** Energy values have been scaled for Malaysian residential context")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🔮 Predict Energy", type="primary", use_container_width=True):
        raw_pred = model.predict(features_df)[0]
        pred = scale_prediction(raw_pred)
        converted, unit = convert_energy_unit(pred, unit_option)
        
        st.subheader("📊 Prediction Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("⚡ Predicted Energy", f"{converted:.2f} {unit}")
        
        if retrofit == 1:
            base_df = features_df.copy()
            base_df['retrofit'] = 0
            raw_base = model.predict(base_df)[0]
            base_pred = scale_prediction(raw_base)
            
            savings = base_pred - pred
            savings_pct = (savings / base_pred) * 100
            savings_conv, _ = convert_energy_unit(savings, unit_option)
            
            m2.metric("💰 Savings", f"{savings_conv:.2f} {unit}", delta=f"{savings_pct:.1f}%")
            m3.metric("🏆 Reduction", f"{savings_pct:.1f}%", delta="Good!")
            
            st.success(f"💡 Retrofit Savings: {savings_conv:.2f} {unit} ({savings_pct:.1f}%)")
            
            # Monthly bill savings
            tariff = 0.52
            monthly_savings = savings * 24 * 30
            monthly_rm = monthly_savings * tariff
            st.info(f"💰 Estimated Monthly Bill Savings: RM {monthly_rm:.2f}/month")
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(8,5))
            ax.bar(['Baseline', 'Retrofitted'], [base_pred, pred], 
                   color=['#e74c3c', '#2ecc71'], edgecolor='black')
            for i, v in enumerate([base_pred, pred]):
                ax.text(i, v + 0.05, f'{v:.2f} kWh', ha='center', fontweight='bold')
            ax.set_ylabel('Energy (kWh)')
            ax.set_title('Retrofit Impact', fontweight='bold')
            st.pyplot(fig)
            
            # Gauge
            fig2, ax2 = plt.subplots(figsize=(8,2.5))
            color = '#2ecc71' if savings_pct > 20 else '#f39c12' if savings_pct > 10 else '#e74c3c'
            label = 'High' if savings_pct > 20 else 'Medium' if savings_pct > 10 else 'Low'
            ax2.barh([0], [min(savings_pct,100)], color=color, height=0.4)
            ax2.barh([0], [100], color='lightgray', height=0.4, alpha=0.3)
            ax2.set_xlim(0,100)
            ax2.set_yticks([])
            ax2.set_xlabel('Savings (%)')
            ax2.set_title(f'Efficiency: {label} ({savings_pct:.1f}%)')
            st.pyplot(fig2)
            
        else:
            retro_df = features_df.copy()
            retro_df['retrofit'] = 1
            raw_retro = model.predict(retro_df)[0]
            retro_pred = scale_prediction(raw_retro)
            potential = pred - retro_pred
            potential_pct = (potential / pred) * 100
            st.info(f"💡 If retrofitted: Save ~{potential:.2f} kWh ({potential_pct:.1f}%)")
            st.caption("👉 Select 'Yes' above to see detailed analysis")

with col2:
    st.info("""
    **📖 About M&V System**
    - **Model:** Random Forest
    - **Features:** Temp, humidity, hour, day, month, area, occupants, retrofit
    - **Malaysia context:** Scaled for residential homes
    - **TNB tariff:** ~RM0.52/kWh
    """)

st.markdown("---")
st.caption("🎓 AI-based M&V System | Thesis Project")
