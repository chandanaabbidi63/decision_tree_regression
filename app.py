# app.py

import streamlit as st
import pickle
import numpy as np
import os

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained model from pickle file."""
    # Try multiple possible paths for the model file
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "model.pkl"),
        os.path.join(os.getcwd(), "model.pkl"),
        "model.pkl"
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
    
    raise FileNotFoundError("model.pkl not found in any expected location")

# Load model with error handling
try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file 'model.pkl' not found. Please ensure the model file exists in the same directory as this app.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Page title
st.title("Decision Tree Regression: Diabetes Progression")
st.caption("Uses the saved model from model.pkl")

st.write("Enter the 10 features and click **Predict**.")

with st.form("regression_form"):
    st.subheader("Input features")

    # Input fields (match model expectations: 10 features)
    age = st.number_input("Age", value=0.05)
    sex = st.number_input("Sex", value=0.05)
    bmi = st.number_input("BMI", value=0.05)
    bp = st.number_input("Blood Pressure", value=0.05)
    s1 = st.number_input("S1", value=0.05)
    s2 = st.number_input("S2", value=0.05)
    s3 = st.number_input("S3", value=0.05)
    s4 = st.number_input("S4", value=0.05)
    s5 = st.number_input("S5", value=0.05)
    s6 = st.number_input("S6", value=0.05)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]], dtype=float)

    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Diabetes Progression: {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
