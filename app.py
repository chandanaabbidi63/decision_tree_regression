import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

st.set_page_config(page_title="Iris Decision Tree Classifier", layout="wide")

st.title("🌸 Iris Decision Tree Classifier")
st.write(
    "Train a Decision Tree on the Iris dataset and interactively predict the species. "
    "Use the controls to tune hyperparameters."
)

# -----------------------------
# Load data
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Model controls")

criterion = st.sidebar.selectbox(
    "Criterion",
    options=["gini", "entropy", "log_loss"],
    index=0,
)

max_depth = st.sidebar.slider(
    "Max depth (0 = None)",
    min_value=0,
    max_value=20,
    value=0,
    step=1,
)

min_samples_split = st.sidebar.slider(
    "Min samples split",
    min_value=2,
    max_value=20,
    value=2,
    step=1,
)

min_samples_leaf = st.sidebar.slider(
    "Min samples leaf",
    min_value=1,
    max_value=20,
    value=1,
    step=1,
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    max_value=10_000,
    value=42,
    step=1,
)

test_size = st.sidebar.slider(
    "Test size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
)

train_button = st.sidebar.button("Train model", type="primary")

# -----------------------------
# Train / evaluate
# -----------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False

if train_button or not st.session_state.trained:
    if max_depth == 0:
        md = None
    else:
        md = int(max_depth)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=md,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(random_state),
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.session_state.model = model
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred
    st.session_state.acc = float(acc)
    st.session_state.cm = cm
    st.session_state.trained = True

# -----------------------------
# Main layout
# -----------------------------
col_left, col_right = st.columns([1, 1.1])

with col_left:
    st.subheader("📊 Evaluation")
    if st.session_state.trained:
        st.metric("Test Accuracy", f"{st.session_state.acc:.3f}")

        st.write("Confusion Matrix")
        fig_cm, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(st.session_state.cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # annotate
        for i in range(st.session_state.cm.shape[0]):
            for j in range(st.session_state.cm.shape[1]):
                ax.text(j, i, str(st.session_state.cm[i, j]), ha="center", va="center", color="black")

        fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig_cm, clear_figure=True)
    else:
        st.info("Click **Train model** to see evaluation results.")

with col_right:
    st.subheader("🌳 Decision Tree")
    if st.session_state.trained:
        show_tree = st.checkbox("Show tree plot (may be large)", value=True)
        if show_tree:
            model = st.session_state.model
            # Use feature names and class names in plot
            fig_tree, ax = plt.subplots(figsize=(12, 7))
            plot_tree(
                model,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                max_depth=None,
                fontsize=8,
                ax=ax,
            )
            st.pyplot(fig_tree, clear_figure=True)
        else:
            st.write("Tree plot hidden.")

# -----------------------------
# Prediction UI
# -----------------------------
st.divider()
st.subheader("🔮 Make a prediction")

feature_inputs = []
with st.container():
    c1, c2 = st.columns(2)
    with c1:
        sl0 = st.slider(feature_names[0], 0.0, 8.0, float(X[:, 0].mean()), step=0.1)
        sl1 = st.slider(feature_names[1], 0.0, 4.5, float(X[:, 1].mean()), step=0.1)
    with c2:
        sl2 = st.slider(feature_names[2], 0.0, 8.0, float(X[:, 2].mean()), step=0.1)
        sl3 = st.slider(feature_names[3], 0.0, 3.5, float(X[:, 3].mean()), step=0.1)

    feature_inputs = [sl0, sl1, sl2, sl3]

predict_button = st.button("Predict", type="primary")

if predict_button:
    if not st.session_state.trained:
        st.warning("Train the model first.")
    else:
        x_in = np.array(feature_inputs, dtype=float).reshape(1, -1)
        pred_idx = int(st.session_state.model.predict(x_in)[0])
        proba = st.session_state.model.predict_proba(x_in)[0]

        st.success(f"Predicted class: **{class_names[pred_idx]}**")

        st.write("Class probabilities")
        prob_col_left, prob_col_right = st.columns(2)
        with prob_col_left:
            st.metric(class_names[0], f"{proba[0]:.3f}")
            st.metric(class_names[1], f"{proba[1]:.3f}")
        with prob_col_right:
            st.metric(class_names[2], f"{proba[2]:.3f}")

        # Also show a compact probability bar chart
        fig_prob, ax = plt.subplots(figsize=(6, 2.5))
        ax.bar(class_names, proba, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Predicted probabilities")
        st.pyplot(fig_prob, clear_figure=True)

