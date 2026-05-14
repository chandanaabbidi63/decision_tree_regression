import streamlit as st
import joblib
import numpy as np
import os

# Load model with caching
@st.cache_resource
def load_model():
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "model.pkl"),
        os.path.join(os.getcwd(), "model.pkl"),
        "model.pkl"
    ]

    for model_path in possible_paths:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model

    raise FileNotFoundError("model.pkl not found")

# Load model
try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file 'model.pkl' not found.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()