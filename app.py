import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open('random_forest_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title("🧠 Brain Tumor Classifier")
st.markdown("This app predicts **Tumor Type** using a Random Forest model trained on brain tumor dataset.")

# Sidebar for user input
st.sidebar.header("Enter Patient Details")

def user_input_features():
    age = st.sidebar.slider("Age", 0, 100, 30)
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    tumor_size = st.sidebar.slider("Tumor Size (cm)", 0.0, 10.0, 5.0)
    location = st.sidebar.selectbox("Tumor Location", ['Frontal', 'Parietal', 'Temporal', 'Occipital'])
    histology = st.sidebar.selectbox("Histology", ['Astrocytoma', 'Glioblastoma', 'Meningioma', 'Medulloblastoma'])
    stage = st.sidebar.selectbox("Tumor Stage", ['I', 'II', 'III', 'IV'])
    chemo = st.sidebar.selectbox("Chemotherapy", ['Yes', 'No'])
    family_history = st.sidebar.selectbox("Family History", ['Yes', 'No'])
    follow_up = st.sidebar.selectbox("Follow-Up Required", ['Yes', 'No'])
    
    data = {
        'Age': age,
        'Gender': gender,
        'Tumor_Size': tumor_size,
        'Location': location,
        'Histology': histology,
        'Stage': stage,
        'Chemotherapy': chemo,
        'Family_History': family_history,
        'Follow_Up_Required': follow_up
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# One-hot encoding or preprocessing logic (update if needed based on your model training)
input_df_processed = pd.get_dummies(input_df)

# Ensure input features match model’s training features
expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else model.get_booster().feature_names
for col in expected_features:
    if col not in input_df_processed.columns:
        input_df_processed[col] = 0

input_df_processed = input_df_processed[expected_features]  # Reorder columns

# Make prediction
prediction = model.predict(input_df_processed)
st.subheader("Predicted Tumor Type")
st.write(prediction[0])
