import joblib
import streamlit as st
import numpy as np
import pandas as pd


@st.cache_resource
def load_artifact():
    return joblib.load("models/decision_tree_diabetes.joblib")


def main():
    st.set_page_config(page_title="Diabetes Regression - Decision Tree", layout="wide")

    st.title("Diabetes Regression (Decision Tree Regressor)")

    try:
        artifact = load_artifact()
    except FileNotFoundError:
        st.error("Model artifact not found. Run `python train.py` first.")
        return

    preprocessing = artifact["preprocessing"]
    model = artifact["model"]
    feature_names = list(artifact["feature_names"])

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Input features")

        inputs = {}
        for name in feature_names:
            # diabetes feature names are numeric features already
            inputs[name] = st.number_input(name, value=0.0, format="%.6f")

        predict_btn = st.button("Predict")

    with cols[1]:
        st.subheader("Prediction")
        if "metrics" in artifact:
            m = artifact["metrics"]
            st.caption(f"Training evaluation: MAE={m['mae']:.3f}, R2={m['r2']:.3f}")

        if predict_btn:
            X = pd.DataFrame([inputs], columns=feature_names)
            X_proc = preprocessing.transform(X)
            y_pred = model.predict(X_proc)
            st.success(f"Predicted disease progression (numeric): **{float(y_pred[0]):.4f}**")


if __name__ == "__main__":
    main()

