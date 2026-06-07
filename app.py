import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stroke Prediction App", layout="centered")
st.title('Stroke Prediction App (Logistic Regression)')
st.write('Enter patient details to predict stroke risk using the Logistic Regression model.')

# Load the deployment bundle
try:
    deployment_bundle_path = '/content/trained_stroke_model_lr.pkl' # Path to your saved model
    deployment_bundle = joblib.load(deployment_bundle_path)
    model = deployment_bundle['model']
    scaler = deployment_bundle['scaler']
    selected_features = deployment_bundle['selected_features']
    optimized_threshold = deployment_bundle['optimized_threshold']
    num_cols_for_scaling = deployment_bundle['num_cols_for_scaling'] # Columns that were scaled
    st.success('Logistic Regression model loaded successfully!')
except FileNotFoundError:
    st.error(f'Error: \'trained_stroke_model_lr.pkl\' not found at {deployment_bundle_path}. Please ensure the model file is in the correct directory in your repository.')
    st.stop()
except Exception as e:
    st.error(f'Error loading model: {e}')
    st.stop()

# Input fields for features
st.header("Patient Information")

# Numerical features
age = st.slider('Age', 0, 100, 50)
avg_glucose_level = st.slider('Average Glucose Level', 50.0, 300.0, 100.0)
bmi = st.slider('BMI', 10.0, 60.0, 25.0)

# Binary features (0 or 1)
hypertension = st.checkbox('Hypertension', False)
heart_disease = st.checkbox('Heart Disease', False)

# Smoking Status (mutually exclusive, choose one)
smoking_status_options = ['never smoked', 'formerly smoked', 'smokes', 'unknown']
selected_smoking_status = st.radio('Smoking Status', smoking_status_options)

# Create a dictionary for the input data, matching the model's expected features
input_data = {
    'age': age,
    'hypertension': 1 if hypertension else 0,
    'heart_disease': 1 if heart_disease else 0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status_formerly smoked': 1 if selected_smoking_status == 'formerly smoked' else 0,
    'smoking_status_never smoked': 1 if selected_smoking_status == 'never smoked' else 0,
    'smoking_status_smokes': 1 if selected_smoking_status == 'smokes' else 0
}

# Convert input data to a DataFrame, ensuring correct feature order
processed_input = pd.DataFrame([input_data])

# Ensure all expected features are present, adding missing ones with 0 if necessary
# This is crucial if some features were dropped during preprocessing (e.g., gender_other) but were part of selected_features
for feature in selected_features:
    if feature not in processed_input.columns:
        processed_input[feature] = 0

processed_input = processed_input[selected_features] # Reorder to match training

# Scale numerical features using the loaded scaler
processed_input_scaled = processed_input.copy()
processed_input_scaled[num_cols_for_scaling] = scaler.transform(processed_input[num_cols_for_scaling])

if st.button('Predict Stroke Risk'):
    # Make prediction
    prediction_proba = model.predict_proba(processed_input_scaled)[:, 1][0]
    prediction_class = (prediction_proba >= optimized_threshold).astype(int)

    st.subheader('Prediction Result:')
    if prediction_class == 1:
        st.error(f'High Risk of Stroke! (Predicted Probability: {prediction_proba:.4f})')
    else:
        st.success(f'Low Risk of Stroke. (Predicted Probability: {prediction_proba:.4f})')

    st.write(f'Decision Threshold: {optimized_threshold:.4f}')
