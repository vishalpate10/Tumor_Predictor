import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open("random_forest_classifier.pkl", "rb"))

# Title
st.set_page_config(page_title="ML Classifier")
st.title("🔍 Machine Learning Prediction App")

# Input fields
st.header("Enter Feature Values")

feature1 = st.number_input("Feature 1", step=0.01)
feature2 = st.number_input("Feature 2", step=0.01)
feature3 = st.number_input("Feature 3", step=0.01)
feature4 = st.number_input("Feature 4", step=0.01)
# Add more input fields below if your model has more features

# Predict button
if st.button("🔮 Predict"):
    try:
        # Prepare input for prediction
        input_features = np.array([[feature1, feature2, feature3, feature4]])
        result = model.predict(input_features)[0]
        st.success(f"✅ Predicted Output: {result}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
