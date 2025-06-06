import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('random_forest_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit page configuration
st.set_page_config(page_title="Brain Tumor Type Predictor", layout="centered")

# Background image CSS - same style as your BMI app
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://www.byoprotein.com/wp-content/uploads/2018/01/fitness-man-desktop-wallpaper-51316-53014-hd-wallpapers.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    /* To make text visible on background, add some transparency */
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.5) !important;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and description inside a container with background overlay for readability
st.title("🧠 Brain Tumor Type Predictor")
st.markdown("Enter patient details below and click **Submit** to predict the tumor type.")

# Form for user inputs
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

# When form is submitted
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

    # One-hot encoding to match training format
    input_data_encoded = pd.get_dummies(input_data)

    # Ensure columns match model training features
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0  # Add missing columns with zero

    input_data_encoded = input_data_encoded[expected_features]  # Reorder columns

    # Predict tumor type (model outputs 0 or 1 or more classes)
    prediction = model.predict(input_data_encoded)

    # Manual mapping of numeric labels to tumor type names
    label_map = {
        0: 'Benign',
        1: 'Malignant',
        2: 'Other'  # Add more if needed based on your model
    }

    predicted_class = label_map.get(prediction[0], 'Unknown')

    # Display result
    st.success(f"🎯 Predicted Tumor Type: **{predicted_class}**")

# Footer with your name and email aligned right
st.markdown("------------")
st.markdown("<div style='text-align: right;'><h4 style='color: white;'>Developed by: Vishal Pate</h4></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: right;'><h4 style='color: white;'>Email: vprakashpate@gmail.com</h4></div>", unsafe_allow_html=True)
