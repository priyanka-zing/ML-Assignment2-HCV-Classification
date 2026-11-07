# HCV Multi-Class Classification

## Dataset
Hepatitis C Virus (HCV) for Egyptian Patients Dataset

## Objective
Predict Baseline Histological Staging using machine learning classification models.

## Models Implemented
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

## Results

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|---------|---------|---------|---------|---------|---------|
| Logistic Regression | 0.2996 | 0.5580 | 0.2988 | 0.2996 | 0.2965 | 0.0642 |
| Decision Tree | 0.2563 | 0.5049 | 0.2566 | 0.2563 | 0.2561 | 0.0082 |
| KNN | 0.2852 | 0.5552 | 0.3016 | 0.2852 | 0.2834 | 0.0519 |
| Naive Bayes | 0.3321 | 0.5683 | 0.3402 | 0.3321 | 0.3291 | 0.1108 |
| SVM | 0.2852 | 0.4747 | 0.2927 | 0.2852 | 0.2831 | 0.0477 |

## Best Performing Model

Naive Bayes achieved the highest Accuracy (33.21%), AUC (0.5683), F1 Score (0.3291), and MCC (0.1108).

## Streamlit Features

- Upload test dataset
- Select machine learning model
- View evaluation metrics
- Visualize confusion matrix
- View classification report
- Compare all models
- Download comparison report

## Project Structure

```text
ML_ASSIGNMENT2_HCV/
├── models/
├── app.py
├── train_models.py
├── README.md
├── requirements.txt
├── HCV-Egy-Data.csv
├── test_data.csv
├── model_comparison.csv
└── .gitignore
```