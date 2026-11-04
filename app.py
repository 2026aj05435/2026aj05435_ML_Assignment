import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt                                              
import seaborn as sns                                                        
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)       

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

st.title("Employee Attrition Prediction")
st.write("Upload the test data CSV to evaluate the trained models.")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

choice = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
model = load_model(MODEL_FILES[choice])
st.sidebar.success(f"Loaded: {choice}")

uploaded = st.file_uploader("Upload test CSV", type="csv")

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.write("Rows:", data.shape[0], " | Columns:", data.shape[1])
    st.dataframe(data.head())

    if "Attrition" not in data.columns:
        st.error("The CSV must contain an 'Attrition' column (the target).")
        st.stop()

    X_up = data.drop(columns=["Attrition"])
    y_up = data["Attrition"]
    if y_up.dtype == object:
        y_up = y_up.map({"Yes": 1, "No": 0})

    pred = model.predict(X_up)
    prob = model.predict_proba(X_up)[:, 1]

    st.subheader(f"Evaluation metrics: {choice}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{accuracy_score(y_up, pred):.4f}")
    c2.metric("AUC", f"{roc_auc_score(y_up, prob):.4f}")
    c3.metric("Precision", f"{precision_score(y_up, pred):.4f}")
    c4.metric("Recall", f"{recall_score(y_up, pred):.4f}")
    c5.metric("F1", f"{f1_score(y_up, pred):.4f}")
    c6.metric("MCC", f"{matthews_corrcoef(y_up, pred):.4f}")

    # confusion matrix (left) and classification report (right)
    left, right = st.columns(2)

    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_up, pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Stays", "Leaves"],
                    yticklabels=["Stayed", "Left"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.subheader("Classification report")
        report = classification_report(y_up, pred,
                                       target_names=["Stayed", "Left"],
                                       output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3))