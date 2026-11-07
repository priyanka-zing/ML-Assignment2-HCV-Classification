# ==================================================
# HCV Multi-Class Classification Model Training
# ==================================================
# This script trains multiple machine learning
# classification models using the HCV Egyptian
# Patients dataset, evaluates their performance,
# and saves trained models for deployment in the
# Streamlit application.
# ==================================================
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
# Load the HCV Egyptian Patients dataset
# containing clinical and laboratory features
# used for disease staging classification.
df = pd.read_csv("HCV-Egy-Data.csv")

# --------------------------------------------------
# Define Features and Target Variable
# --------------------------------------------------
# Target variable:
# Baseline Histological Staging
#
# Features:
# All remaining clinical attributes.
target_column = "Baselinehistological staging"

X = df.drop(target_column, axis=1)
y = df[target_column]

# --------------------------------------------------
# Split Dataset into Training and Testing Sets
# --------------------------------------------------
# Use 80% of records for training and 20%
# for model evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# Standardize Feature Values
# --------------------------------------------------
# Feature scaling is required for distance-based
# and optimization-based algorithms such as
# Logistic Regression, KNN and SVM.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Create Directory for Trained Models
# --------------------------------------------------
# Store trained models and preprocessing
# artifacts used during deployment.

os.makedirs("models", exist_ok=True)

# Save fitted scaler for reuse in Streamlit app
joblib.dump(scaler, "models/scaler.pkl")

# --------------------------------------------------
# Train Logistic Regression Model
# --------------------------------------------------
# Baseline linear classification model.
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
joblib.dump(lr, "models/logistic.pkl")

# --------------------------------------------------
# Train Decision Tree Classifier
# --------------------------------------------------
# Tree-based model used for interpretable
# classification.
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
joblib.dump(dt, "models/decision_tree.pkl")

# --------------------------------------------------
# Train K-Nearest Neighbors Model
# --------------------------------------------------
# Instance-based learning algorithm using
# nearest neighbor similarity.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
joblib.dump(knn, "models/knn.pkl")

# --------------------------------------------------
# Train Naive Bayes Classifier
# --------------------------------------------------
# Probabilistic classification algorithm.
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
joblib.dump(nb, "models/naive_bayes.pkl")

# --------------------------------------------------
# Train Support Vector Machine Model
# --------------------------------------------------
# Margin-based classifier using optimized
# decision boundaries.
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)
joblib.dump(svm, "models/svm.pkl")

# --------------------------------------------------
# Save Test Dataset
# --------------------------------------------------
# Export testing records to support model
# evaluation in the Streamlit application.
test_data = pd.concat(
    [pd.DataFrame(X_test), pd.DataFrame(y_test)],
    axis=1
)

test_data.to_csv(
    "test_data.csv",
    index=False
)

print("Models trained successfully!")
print("Test data saved successfully!")

# --------------------------------------------------
# Model Evaluation Function
# --------------------------------------------------
# Calculates Accuracy, AUC, Precision,
# Recall, F1 Score and MCC for each model.
def evaluate_model(model, X_test, y_test, model_name):

# Generate predictions using trained model
    y_pred = model.predict(X_test)

# Compute model performance metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

# Calculate multiclass AUC score using
# One-vs-Rest strategy
    try:
        y_prob = model.predict_proba(X_test)

        classes = sorted(y.unique())

        y_test_bin = label_binarize(
            y_test,
            classes=classes
        )

        auc = roc_auc_score(
            y_test_bin,
            y_prob,
            multi_class="ovr"
        )

    except:
        auc = "N/A"

    return [
        model_name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ]

# --------------------------------------------------
# Evaluate All Trained Models
# --------------------------------------------------
# Store performance metrics of each model
# for comparison and reporting.
results = []

results.append(
    evaluate_model(
        lr,
        X_test_scaled,
        y_test,
        "Logistic Regression"
    )
)

results.append(
    evaluate_model(
        dt,
        X_test,
        y_test,
        "Decision Tree"
    )
)

results.append(
    evaluate_model(
        knn,
        X_test_scaled,
        y_test,
        "KNN"
    )
)

results.append(
    evaluate_model(
        nb,
        X_test_scaled,
        y_test,
        "Naive Bayes"
    )
)

results.append(
    evaluate_model(
        svm,
        X_test_scaled,
        y_test,
        "SVM"
    )
)


# Create comparison table containing
# evaluation results of all classifiers
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]
)

print("\nModel Comparison Report")
print(results_df)

# Save model comparison report for
# analysis and Streamlit visualization
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nmodel_comparison.csv generated successfully!")