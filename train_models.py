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

# Load Dataset
df = pd.read_csv("HCV-Egy-Data.csv")

# Define Target
target_column = "Baselinehistological staging"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale Features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
joblib.dump(lr, "models/logistic.pkl")

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
joblib.dump(dt, "models/decision_tree.pkl")

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
joblib.dump(knn, "models/knn.pkl")

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
joblib.dump(nb, "models/naive_bayes.pkl")

# SVM
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)
joblib.dump(svm, "models/svm.pkl")

# Save Test Data
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

def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

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

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nmodel_comparison.csv generated successfully!")