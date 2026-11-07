import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")

    with col2:
        st.metric("AUC", f"{auc:.4f}")

    with col3:
        st.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Recall", f"{recall:.4f}")

    with col5:
        st.metric("F1 Score", f"{f1:.4f}")

    with col6:
        st.metric("MCC", f"{mcc:.4f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y,
        y_pred
    )

    st.text(report)

    st.subheader("Model Comparison Results")

    comparison_df = pd.read_csv("model_comparison.csv")

    st.dataframe(comparison_df)

    best_model = comparison_df.loc[
        comparison_df["Accuracy"].idxmax()
    ]

    st.success(
        f"Best Model: {best_model['Model']} "
        f"(Accuracy = {best_model['Accuracy']:.4f})"
    )

    st.download_button(
        label="Download Model Comparison Report",
        data=comparison_df.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv"
    )