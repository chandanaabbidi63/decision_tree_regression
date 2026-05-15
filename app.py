# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Prediction - SVC",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: #0e4c92;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: gray;
}

.stButton>button {
    background-color: #0e4c92;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    '<p class="title">🩺 Breast Cancer Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Support Vector Classification (SVC) using Machine Learning</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

cancer = load_breast_cancer()

X = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

y = pd.Series(cancer.target)

# ---------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------

st.header("📂 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Features Dataset")
    st.dataframe(X.head())

with col2:
    st.subheader("Target Dataset")
    st.dataframe(y.head())

st.write("### Dataset Shape")
st.write("Features Shape:", X.shape)
st.write("Target Shape:", y.shape)

# ---------------------------------------------------
# DATA VISUALIZATION
# ---------------------------------------------------

st.header("📊 Data Visualization")

fig1, ax1 = plt.subplots(figsize=(15, 6))

sns.boxplot(data=X.iloc[:, :10], ax=ax1)

plt.xticks(rotation=90)

st.pyplot(fig1)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING (Important for SVC)
# ---------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# MODEL CREATION - SVC
# ---------------------------------------------------

st.header("🤖 Support Vector Classifier (SVC) Model")

st.info("ℹ️ **Note:** Feature scaling (StandardScaler) is applied before training SVC for better performance.")

model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)

# Train Model
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------
# MODEL PREDICTIONS
# ---------------------------------------------------

y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------

st.header("📈 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{accuracy:.2f}")

with col2:
    st.metric("Training Samples", len(X_train))

with col3:
    st.metric("Testing Samples", len(X_test))

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig2, ax2 = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=cancer.target_names,
    yticklabels=cancer.target_names,
    ax=ax2
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig2)

# ---------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=cancer.target_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# ---------------------------------------------------
# SVC MODEL DETAILS
# ---------------------------------------------------

st.subheader("ℹ️ SVC Model Details")

model_details = pd.DataFrame({
    "Parameter": ["Kernel", "C (Regularization)", "Gamma", "Probability"],
    "Value": ["RBF (Radial Basis Function)", "1.0", "scale", "True"]
})

st.dataframe(model_details)

# ---------------------------------------------------
# USER INPUT SECTION
# ---------------------------------------------------

st.header("🔍 Predict Breast Cancer")

st.write("Enter the feature values below:")

input_data = []

col1, col2 = st.columns(2)

for i, feature in enumerate(cancer.feature_names):

    mean_value = float(X[feature].mean())

    if i % 2 == 0:

        with col1:

            value = st.number_input(
                feature,
                value=mean_value,
                format="%.4f"
            )

    else:

        with col2:

            value = st.number_input(
                feature,
                value=mean_value,
                format="%.4f"
            )

    input_data.append(value)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("Predict Cancer Type"):

    input_array = np.array(input_data).reshape(1, -1)

    # Scale the input using the same scaler used during training
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    probability = model.predict_proba(input_array_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 0:

        st.error("⚠️ Malignant Cancer Detected")

    else:

        st.success("✅ Benign Cancer Detected")

    # Probability Table
    st.write("### Prediction Probability")

    prob_df = pd.DataFrame({
        "Class": ["Malignant", "Benign"],
        "Probability": probability[0]
    })

    st.dataframe(prob_df)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <center>
        <h4>
            Developed using Streamlit & Scikit-Learn | SVC Model
        </h4>
    </center>
    """,
    unsafe_allow_html=True
)
