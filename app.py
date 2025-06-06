import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1️⃣ Set Streamlit page configuration FIRST
st.set_page_config(page_title="Brain Tumor Type Predictor", layout="centered")

# 2️⃣ Add background image using HTML and CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1581091012184-5c4f44c19d63?auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3️⃣ Load trained model
with open('random_forest_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# 4️⃣ App title
st.markdown("<div class='block-container'>", unsafe_allow_html=True)
st.title("🧠 Brain Tumor Type Predictor")
st.markdown("Enter patient details below and click **Submit** to predict the tumor type.")

# 5️⃣ Input form
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

# 6️⃣ Handle submission
if submit_button:
    # Create DataFrame from input
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

    # One-hot encode to match training data
    input_encoded = pd.get_dummies(input_data)

    # Ensure feature alignment with training
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[expected_features]

    # Predict
    prediction = model.predict(input_encoded)

    # Map prediction to label
    label_map = {
        0: 'Benign',
        1: 'Malignant',
        2: 'Other'  # If applicable
    }

    predicted_class = label_map.get(prediction[0], 'Unknown')

    # Display result
    if predicted_class == 'Unknown':
        st.warning("Prediction label not recognized. Please check the model or label mapping.")
    else:
        st.success(f"🎯 Predicted Tumor Type: **{predicted_class}**")

st.markdown("</div>", unsafe_allow_html=True)
