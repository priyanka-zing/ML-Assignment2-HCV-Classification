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