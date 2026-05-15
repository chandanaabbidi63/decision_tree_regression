# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction - Decision Tree Regression",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🧠 Diabetes Prediction using Decision Tree Regression")
st.markdown("Built with Streamlit + Scikit-learn")

st.markdown("---")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

diabetes = load_diabetes()

X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

df = X.copy()
df["target"] = y

# ---------------------------------------------------
# DATA OVERVIEW
# ---------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("### Features")
    st.dataframe(X.head())

with col2:
    st.write("### Target")
    st.dataframe(y.head())

st.write("Shape:", df.shape)

# ---------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------

st.header("📉 Data Visualization")

fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.boxplot(data=X.iloc[:, :5], ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

st.header("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R² Score", f"{r2:.2f}")

# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------

st.subheader("📊 Actual vs Predicted")

fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred)
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
st.pyplot(fig2)

# ---------------------------------------------------
# DECISION TREE VISUALIZATION
# ---------------------------------------------------

st.header("🌳 Decision Tree Structure")

fig3, ax3 = plt.subplots(figsize=(18, 8))

plot_tree(
    model,
    feature_names=diabetes.feature_names,
    filled=True,
    fontsize=7,
    ax=ax3
)

st.pyplot(fig3)

# ---------------------------------------------------
# USER INPUT PREDICTION
# ---------------------------------------------------

st.header("🔮 Predict Diabetes Progression")

st.write("Enter feature values:")

input_data = []

cols = st.columns(5)

for i, feature in enumerate(diabetes.feature_names):
    value = cols[i % 5].number_input(
        feature,
        value=float(X[feature].mean())
    )
    input_data.append(value)

if st.button("Predict"):

    input_array = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_array)

    st.success(f"Predicted Disease Progression: {prediction[0]:.2f}")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.markdown("### 🚀 Built with Streamlit + Scikit-learn")