import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "decision_tree_diabetes.joblib")


@st.cache_resource
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model artifact not found at: {MODEL_PATH}.\n"
            f"Run `python train.py` first."
        )
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="Diabetes Regression - Decision Tree", layout="wide")
    st.title("Diabetes Regression (Decision Tree Regressor)")

    try:
        artifact = load_artifact()
    except Exception as e:
        st.error(str(e))
        return

    preprocessing = artifact["preprocessing"]
    model = artifact["model"]
    feature_names = list(artifact["feature_names"])

    left, right = st.columns(2)

    with left:
        st.subheader("Input features")
        inputs = {}
        for name in feature_names:
            inputs[name] = st.number_input(name, value=0.0, format="%.6f")

        predict_btn = st.button("Predict")

    with right:
        st.subheader("Prediction")
        if "metrics" in artifact:
            m = artifact["metrics"]
            st.caption(f"Training evaluation: MAE={m['mae']:.3f}, R2={m['r2']:.3f}")

        if predict_btn:
            X = pd.DataFrame([inputs], columns=feature_names)
            X_proc = preprocessing.transform(X)
            y_pred = model.predict(X_proc)
            st.success(f"Predicted disease progression: **{float(y_pred[0]):.4f}**")


if __name__ == "__main__":
    main()

