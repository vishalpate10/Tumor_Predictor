import streamlit as st
import pandas as pd
import pickle

# Load model
with open("random_forest_classifier.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧠 Brain Tumor Type Predictor")

st.markdown("Enter the patient's details:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0, value=15.0)
survival_rate = st.number_input("Estimated Survival Rate (%)", min_value=0.0, max_value=100.0, value=80.0)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
location = st.selectbox("Tumor Location", ["Frontal", "Parietal", "Temporal", "Occipital", "Cerebellum", "Other"])

# Fake encoders for demo (you can replace with actual encoding)
def encode_input(gender, location):
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    location_map = {"Frontal": 0, "Parietal": 1, "Temporal": 2, "Occipital": 3, "Cerebellum": 4, "Other": 5}
    return gender_map.get(gender, -1), location_map.get(location, -1)

gender_encoded, location_encoded = encode_input(gender, location)

# Validate inputs
if gender_encoded == -1 or location_encoded == -1:
    st.error("❌ Invalid categorical input provided.")
else:
    input_df = pd.DataFrame([{
        "Age": age,
        "Tumor_Size": tumor_size,
        "Survival_Rate": survival_rate,
        "Gender": gender_encoded,
        "Tumor_Location": location_encoded
    }])

    if st.button("Predict Tumor Type"):
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted Tumor Type: **{prediction}**")
