import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

st.title("Employee Attrition Prediction")
st.write("Upload the test data CSV to evaluate the trained models.")

# Model name -> saved file
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

# Load a model once and remember it
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Dropdown in the sidebar
choice = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
model = load_model(MODEL_FILES[choice])
st.sidebar.success(f"Loaded: {choice}")

# CSV upload and preview
uploaded = st.file_uploader("Upload test CSV", type="csv")

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.write("Rows:", data.shape[0], " | Columns:", data.shape[1])
    st.dataframe(data.head())