import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Load models and scaler
# -----------------------------
log_reg = joblib.load("model/logistic_regression.pkl")
decision_tree = joblib.load("model/decision_tree.pkl")
knn = joblib.load("model/knn.pkl")
naive_bayes = joblib.load("model/naive_bayes.pkl")
random_forest = joblib.load("model/random_forest.pkl")
xgboost = joblib.load("model/xgboost.pkl")

# Load the fitted scaler saved from notebook
scaler = joblib.load("model/scaler.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("ML Assignment 2 - Breast Cancer Classification")

# Dropdown FIRST
model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

# File upload AFTER dropdown
uploaded_file = st.file_uploader("Upload test CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    # Separate features and target
    X_test_ext = data.drop("target", axis=1)
    y_test_ext = data["target"]

    # Apply the SAME scaler
    X_test_ext_scaled = scaler.transform(X_test_ext)

    # Pick model based on dropdown
    if model_choice == "Logistic Regression":
        model = log_reg
    elif model_choice == "Decision Tree":
        model = decision_tree
    elif model_choice == "kNN":
        model = knn
    elif model_choice == "Naive Bayes":
        model = naive_bayes
    elif model_choice == "Random Forest":
        model = random_forest
    else:
        model = xgboost

    # Predictions
    y_pred = model.predict(X_test_ext_scaled)
    y_prob = model.predict_proba(X_test_ext_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test_ext, y_pred),
        "Precision": precision_score(y_test_ext, y_pred),
        "Recall": recall_score(y_test_ext, y_pred),
        "F1": f1_score(y_test_ext, y_pred),
        "MCC": matthews_corrcoef(y_test_ext, y_pred),
        "AUC": roc_auc_score(y_test_ext, y_prob) if y_prob is not None else np.nan
    }

    st.subheader("Evaluation Metrics")
    for k, v in metrics.items():
        st.write(f"{k}: {v:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test_ext, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)