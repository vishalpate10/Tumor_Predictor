import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('random_forest_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit page setup
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Title
st.title("🧠 Brain Tumor Type Predictor")
st.markdown("Enter patient details below and click **Submit** to predict the tumor type.")

# Input fields on main page
with st.form(key='tumor_form'):
    age = st.slider("Age", 0, 100, 30)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    tumor_size = st.slider("Tumor Size (in cm)", 0.0, 10.0, 5.0)
    location = st.selectbox("Tumor Location", ['Frontal', 'Parietal', 'Temporal', 'Occipital'])
    histology = st.selectbox("Histology", ['Astrocytoma', 'Glioblastoma', 'Meningioma', 'Medulloblastoma'])
    stage = st.selectbox("Tumor Stage", ['I', 'II', 'III', 'IV'])
    chemotherapy = st.selectbox("Chemotherapy", ['Yes', 'No'])
    family_history = st.selectbox("Family History", ['Yes', 'No'])
    follow_up = st.selectbox("Follow-Up Required", ['Yes', 'No'])

    # Submit button
    submit_button = st.form_submit_button(label='Submit')

# When user clicks Submit
if submit_button:
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Tumor_Size': [tumor_size],
        'Location': [location],
        'Histology': [histology],
        'Stage': [stage],
        'Chemotherapy': [chemotherapy],
        'Family_History': [family_history],
        'Follow_Up_Required': [follow_up]
    })

    # Preprocess input data (one-hot encoding)
    input_data_encoded = pd.get_dummies(input_data)

    # Ensure all expected features exist
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0  # Add missing columns with 0

    input_data_encoded = input_data_encoded[expected_features]  # Reorder to match model

    # Predict
    prediction = model.predict(input_data_encoded)

    # Display result
    st.success(f"🎯 Predicted Tumor Type: **{prediction[0]}**")
