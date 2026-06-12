import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the deployment bundle
@st.cache_resource
def load_model():
    return joblib.load('trained_stroke_model_lr.pkl')

deployment_bundle = load_model()
model = deployment_bundle['model']
scaler = deployment_bundle['scaler']
selected_features = deployment_bundle['selected_features']
optimized_threshold = deployment_bundle['optimized_threshold']

st.title('Stroke Risk Prediction App')
st.write('Enter patient details to predict stroke risk.')

# Input features from the user
age = st.slider('Age', 0, 120, 50)
avg_glucose_level = st.slider('Average Glucose Level', 50.0, 300.0, 100.0)
bmi = st.slider('BMI', 10.0, 60.0, 25.0)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([[age, avg_glucose_level, bmi]], columns=selected_features)

# Scale the input data using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction_proba = model.predict_proba(input_data_scaled)[:, 1]

# Display result based on optimized threshold
if st.button('Predict'):
    if prediction_proba >= optimized_threshold:
        st.error(f'High Risk of Stroke (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'Low Risk of Stroke (Probability: {prediction_proba[0]:.2f})')