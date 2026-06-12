import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the deployment bundle (model, scaler, selected features, optimized threshold)
@st.cache_resource
def load_model():
    return joblib.load('trained_stroke_model_lr.pkl')

deployment_bundle = load_model()
model = deployment_bundle['model']
scaler = deployment_bundle['scaler']
selected_features = deployment_bundle['selected_features']
optimized_threshold = deployment_bundle['optimized_threshold']

st.title('☥️ Stroke Risk Prediction Portal')
st.markdown("""
This application uses a clinical Logistic Regression model to estimate the probability of stroke 
based on key health metrics: **Age, Glucose Levels, and BMI**.
""")

st.sidebar.header('Patient Diagnostics')

# Input features from the user via sidebar
age = st.sidebar.slider('Age', 0, 120, 50)
avg_glucose_level = st.sidebar.slider('Average Glucose Level (mg/dL)', 50.0, 300.0, 100.0)
bmi = st.sidebar.slider('Body Mass Index (BMI)', 10.0, 60.0, 25.0)

# Prepare input data for prediction
input_data = pd.DataFrame([[age, avg_glucose_level, bmi]], columns=selected_features)

# Scale the input data using the pre-fitted scaler
input_data_scaled = scaler.transform(input_data)

# Make prediction probability
prediction_proba = model.predict_proba(input_data_scaled)[:, 1][0]

st.subheader('Prediction Result')
if st.button('Run Diagnostic'):
    if prediction_proba >= optimized_threshold:
        st.error(f'**Result: High Risk**')
        st.write(f'Calculated Probability: {prediction_proba:.2%}')
        st.warning('The score exceeds the clinical threshold. Preventive consultation is recommended.')
    else:
        st.success(f'**Result: Low Risk**')
        st.write(f'Calculated Probability: {prediction_proba:.2%}')
        st.info('The score is within the acceptable range based on current clinical data.')

st.divider()
st.caption('Note: This tool is for educational purposes and based on the Stroke Risk Dataset analysis.')
