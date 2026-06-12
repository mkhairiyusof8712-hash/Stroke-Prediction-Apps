import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stroke Prediction App", layout="centered")
st.title('Stroke Prediction App (Logistic Regression)')
st.write('Enter patient details to predict stroke risk using the Logistic Regression model.')

# Load the deployment bundle
try:
    deployment_bundle_path = 'trained_stroke_model_lr.pkl' # Path to your saved model
    deployment_bundle = joblib.load(deployment_bundle_path)
    model = deployment_bundle['model']
    scaler = deployment_bundle['scaler']
    # Use the selected_features from the deployment bundle to ensure consistency
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

input_data_raw = {}

# Define ranges for numerical features (adjust as needed)
numerical_ranges = {
    'age': (0, 100, 50),
    'avg_glucose_level': (50.0, 300.0, 100.0),
    'bmi': (10.0, 60.0, 25.0)
}

# Handle numerical inputs
for feature_name in num_cols_for_scaling:
    min_val, max_val, default_val = numerical_ranges.get(feature_name, (0.0, 1.0, 0.5)) # Default generic if not found
    input_data_raw[feature_name] = st.slider(feature_name.replace('_', ' ').title(), min_val, max_val, default_val)

# Handle smoking status as a single selectbox, then one-hot encode it
smoking_options = ['never smoked', 'smokes', 'formerly smoked', 'unknown']
selected_smoking_status = st.selectbox('Smoking Status', smoking_options, index=0) # Default to 'never smoked'

# Initialize one-hot encoded smoking status features to 0
input_data_raw['smoking_status_never smoked'] = 0
input_data_raw['smoking_status_smokes'] = 0
input_data_raw['smoking_status_unknown'] = 0

# Set the selected smoking status feature to 1
if selected_smoking_status == 'never smoked':
    input_data_raw['smoking_status_never smoked'] = 1
elif selected_smoking_status == 'smokes':
    input_data_raw['smoking_status_smokes'] = 1
elif selected_smoking_status == 'unknown':
    input_data_raw['smoking_status_unknown'] = 1
# 'formerly smoked' is the base case due to drop_first=True, so no specific _formerly_smoked column is set to 1.

# Handle binary features (hypertension, heart_disease)
binary_features = []
for f in selected_features:
    if f not in num_cols_for_scaling and not f.startswith('smoking_status_'):
        binary_features.append(f)

for feature_name in binary_features:
    input_data_raw[feature_name] = st.checkbox(feature_name.replace('_', ' ').title(), value=False) # Default to No (0)
    input_data_raw[feature_name] = 1 if input_data_raw[feature_name] else 0 # Convert bool to int

# Convert input data to a DataFrame, ensuring correct feature order
processed_input = pd.DataFrame([input_data_raw])

# Ensure all selected_features are present and in the correct order
processed_input = processed_input[selected_features]



# Scale numerical features using the loaded scaler
processed_input_scaled = processed_input.copy()
if num_cols_for_scaling:
    processed_input_scaled[num_cols_for_scaling] = scaler.transform(processed_input[num_cols_for_scaling].values)

if st.button('Predict Stroke Risk'):
    # Make prediction
    prediction_proba = model.predict_proba(processed_input_scaled.values)[:, 1][0]
    prediction_class = (prediction_proba >= optimized_threshold).astype(int)

    st.subheader('Prediction Result:')
    if prediction_class == 1:
        st.error(f'High Risk of Stroke! (Predicted Probability: {prediction_proba:.4f})')
    else:
        st.success(f'Low Risk of Stroke. (Predicted Probability: {prediction_proba:.4f})')

    st.write(f'Decision Threshold: {optimized_threshold:.4f}')
