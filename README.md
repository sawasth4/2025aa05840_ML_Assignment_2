# ML Assignment 2 – Breast Cancer Classification

## Problem Statement
The objective of this assignment is to implement multiple machine learning classification models on a chosen dataset, evaluate their performance using standard metrics, and deploy an interactive Streamlit web application. The app allows users to upload test data, select models, and view evaluation results in real time.

## Dataset Description
We used the **Breast Cancer Wisconsin (Diagnostic) dataset** from the UCI repository.  
- **Instances:** 569 samples  
- **Features:** 30 numeric features (e.g., mean radius, texture, concavity)  
- **Target:** Binary classification (Malignant = 1, Benign = 0)  
- The dataset is well-balanced and widely used for benchmarking classification models.

## Models Used
The following six models were implemented:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN)  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

## Performance Comparison Table

| ML Model Name       | Accuracy | AUC    | Precision | Recall  | F1     | MCC    |
|---------------------|----------|--------|-----------|---------|--------|--------|
| Logistic Regression | 0.9825   | 0.9954 | 0.9861    | 0.9861  | 0.9861 | 0.9623 |
| Decision Tree       | 0.9123   | 0.9157 | 0.9559    | 0.9028  | 0.9286 | 0.8174 |
| kNN                 | 0.9561   | 0.9788 | 0.9589    | 0.9722  | 0.9655 | 0.9054 |
| Naive Bayes         | 0.9298   | 0.9868 | 0.9444    | 0.9444  | 0.9444 | 0.8492 |
| Random Forest       | 0.9561   | 0.9939 | 0.9589    | 0.9722  | 0.9655 | 0.9054 |
| XGBoost             | 0.9561   | 0.9901 | 0.9467    | 0.9861  | 0.9660 | 0.9058 |

## Observations

| ML Model Name       | Observation about model performance |
|---------------------|--------------------------------------|
| Logistic Regression | Achieved the highest accuracy and AUC; strong generalization and low overfitting. |
| Decision Tree       | Lower accuracy compared to others; prone to variance, but precision remains high. |
| kNN                 | Balanced performance with high recall; effective on this dataset. |
| Naive Bayes         | Moderate accuracy but very strong AUC; simple and efficient baseline model. |
| Random Forest       | Excellent performance, nearly matching Logistic Regression; robust ensemble method. |
| XGBoost             | High recall and strong AUC; competitive with Random Forest, slightly better F1. |

## Live App
The deployed Streamlit app can be accessed here:  
👉 [Click to open the app](https://2025aa05840mlassignment2-xhsnudkcp86vtdjoenhytf.streamlit.app/)

## Features of the Streamlit App
- CSV dataset upload option (test data only).  
- Model selection dropdown.  
- Display of evaluation metrics (Accuracy, Precision, Recall, F1, MCC, AUC).  
- Confusion matrix visualization.  

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd project-folder
