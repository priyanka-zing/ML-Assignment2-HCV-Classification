import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import (
    label_binarize
)

from sklearn.metrics import roc_auc_score

st.set_page_config(
    page_title="HCV Classification",
    layout="wide"
)

st.title("HCV Multi-Class Classification")

st.write(
    "Evaluate saved machine learning models using uploaded test data."
)

uploaded_file = st.file_uploader(
    "Upload Test Data CSV",
    type=["csv"]
)

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]
)

model_map = {
    "Logistic Regression":
        "models/logistic.pkl",

    "Decision Tree":
        "models/decision_tree.pkl",

    "KNN":
        "models/knn.pkl",

    "Naive Bayes":
        "models/naive_bayes.pkl",

    "SVM":
        "models/svm.pkl"
}

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(data.head())

    target_column = "Baselinehistological staging"

    X = data.drop(
        target_column,
        axis=1
    )

    y = data[target_column]

    model = joblib.load(
        model_map[model_choice]
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    if model_choice in [
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]:

        X_test = scaler.transform(X)

    else:

        X_test = X


    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y,
        y_pred
    )

    precision = precision_score(
        y,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y,
        y_pred,
        average="weighted"
    )

    mcc = matthews_corrcoef(
        y,
        y_pred
    )

    try:

        y_prob = model.predict_proba(
            X_test
        )

        classes = sorted(
            y.unique()
        )

        y_bin = label_binarize(
            y,
            classes=classes
        )

        auc = roc_auc_score(
            y_bin,
            y_prob,
            multi_class="ovr"
        )

    except:

        auc = "N/A"

        st.subheader("Evaluation Metrics")

    metrics_df = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ],

        "Value": [
            accuracy,
            auc,
            precision,
            recall,
            f1,
            mcc
        ]
    })

    st.table(metrics_df)

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y,
        y_pred
    )

    st.write(cm)

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y,
        y_pred
    )

    st.text(report)