import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the deployment bundle
@st.cache_resource
def load_model():
    # This file contains: model, scaler, selected_features, and optimized_threshold
    return joblib.load('trained_stroke_model_lr.pkl')

try:
    bundle = load_model()
    model = bundle['model']
    scaler = bundle['scaler']
    features = bundle['selected_features']
    threshold = bundle['optimized_threshold']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.set_page_config(page_title="Stroke Risk Portal", page_icon="🏥")
st.title('🏥 Stroke Risk Prediction Portal')

st.markdown("""
This diagnostic tool uses a **Logistic Regression** model trained on clinical data 
to estimate stroke probability based on Age, Glucose, and BMI.
""")

st.sidebar.header('Patient Clinical Metrics')

# User Inputs
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=50)
glucose = st.sidebar.number_input('Avg Glucose Level (mg/dL)', min_value=40.0, max_value=300.0, value=100.0)
bmi = st.sidebar.number_input('Body Mass Index (BMI)', min_value=10.0, max_value=60.0, value=25.0)

# Processing
if st.button('Run Risk Analysis'):
    # Create DataFrame matching training feature names
    raw_input = pd.DataFrame([[age, glucose, bmi]], columns=features)
    
    # IMPORTANT: Use the scaler from the bundle to transform input
    scaled_input = scaler.transform(raw_input)
    
    # Get probability for class 1 (Stroke)
    prob = model.predict_proba(scaled_input)[0, 1]
    
    st.divider()
    st.subheader("Diagnostic Result")
    
    col1, col2 = st.columns(2)
    col1.metric("Calculated Risk", f"{prob:.2%}")
    col2.metric("Clinical Threshold", f"{float(threshold):.2%}")

    if prob >= threshold:
        st.error("⚠️ **Result: HIGH RISK**")
        st.warning("The patient's profile exceeds the optimized clinical threshold for stroke risk. Immediate medical consultation is advised.")
    else:
        st.success("✅ **Result: LOW RISK**")
        st.info("The patient's profile is currently below the high-risk threshold. Continue regular health monitoring.")

    with st.expander("View Technical Details"):
        st.write("**Feature Scaling Verification (Internal):**")
        st.write(pd.DataFrame(scaled_input, columns=[f"scaled_{c}" for c in features]))

st.divider()
st.caption("Disclaimer: This is a decision support tool for educational use only and does not replace professional medical advice.")
