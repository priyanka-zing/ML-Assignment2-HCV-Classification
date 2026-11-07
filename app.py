# --------------------------------------------------
# Import Required Libraries
# --------------------------------------------------
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

# --------------------------------------------------
# Configure Streamlit Dashboard
# --------------------------------------------------
# Set page title and layout for better user experience

st.set_page_config(
    page_title="HCV Classification",
    layout="wide"
)

# --------------------------------------------------
# Dashboard Title and Description
# --------------------------------------------------
# Display application title and project overview

st.title("HCV Multi-Class Classification")

st.write(
    "Evaluate saved machine learning models using uploaded test data."
)

# --------------------------------------------------
# Dataset Upload and Model Selection
# --------------------------------------------------
# Allow users to upload test dataset and choose
# the machine learning model for evaluation

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

# --------------------------------------------------
# Map User Selection to Saved Model Files
# --------------------------------------------------
# Each dropdown option corresponds to a trained
# model stored in the models directory

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

# --------------------------------------------------
# Process Uploaded Dataset
# --------------------------------------------------
# Continue only when a valid CSV file is uploaded

if uploaded_file is not None:

# Load the uploaded CSV file and display
# the first few records for verification
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(data.head())

# Separate input features and target variable
# Target: Histological staging classification
    target_column = "Baselinehistological staging"

    X = data.drop(
        target_column,
        axis=1
    )

    y = data[target_column]

# Load selected trained model from disk
    model = joblib.load(
        model_map[model_choice]
    )

# Load scaler used during model training
# to ensure consistent feature preprocessing
    scaler = joblib.load(
        "models/scaler.pkl"
    )

# Apply feature scaling for models that require
# normalized input data
    if model_choice in [
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]:

        X_test = scaler.transform(X)

    else:

        X_test = X

# Generate predictions using selected model
    y_pred = model.predict(X_test)

# Calculate performance evaluation metrics
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

# Compute multiclass AUC score using
# One-vs-Rest strategy
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


# --------------------------------------------------
# Display Model Performance Metrics
# --------------------------------------------------
# Show evaluation metrics using KPI cards

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

# --------------------------------------------------
# Visualize Prediction Performance
# --------------------------------------------------
# Display confusion matrix as heatmap

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

# Display precision, recall and F1-score
# for each target class

    st.subheader("Classification Report")

    report = classification_report(
        y,
        y_pred
    )

    st.text(report)

# --------------------------------------------------
# Compare Performance Across All Models
# --------------------------------------------------
# Load previously generated comparison report

    st.subheader("Model Comparison Results")

    comparison_df = pd.read_csv("model_comparison.csv")

    st.dataframe(comparison_df)

# Identify model with highest accuracy
    best_model = comparison_df.loc[
        comparison_df["Accuracy"].idxmax()
    ]

    st.success(
        f"Best Model: {best_model['Model']} "
        f"(Accuracy = {best_model['Accuracy']:.4f})"
    )

# Allow users to download model comparison
# results as CSV file
    st.download_button(
        label="Download Model Comparison Report",
        data=comparison_df.to_csv(index=False),
        file_name="model_comparison.csv",
        mime="text/csv"
    )