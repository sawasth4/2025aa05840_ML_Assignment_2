# ML Assignment 2 – Breast Cancer Classification

## Problem Statement
The aim of this assignment is to design, implement, and evaluate multiple machine learning classification models on a biomedical dataset. The task involves building an interactive Streamlit web application that allows users to upload test data, select models, and view evaluation metrics. The project demonstrates an end-to-end ML workflow: data preprocessing, model training, evaluation, UI development, and deployment on Streamlit Community Cloud.

## Dataset Description
We selected the **Breast Cancer Wisconsin (Diagnostic) dataset** from the UCI Machine Learning Repository.  
- **Instances:** 569 patient records  
- **Features:** 30 numeric attributes describing cell nuclei characteristics (e.g., radius, texture, concavity, symmetry)  
- **Target Variable:** Binary classification — Malignant (1) vs. Benign (0)  
- **Rationale:** The dataset is widely used for benchmarking classification algorithms, contains sufficient features (>12), and meets the minimum instance requirement (>500). Its balanced distribution ensures reliable evaluation across models.

## Models Implemented
Six classification models were trained and evaluated on the dataset:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN)  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

## Performance Comparison

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
| Logistic Regression | Achieved the highest accuracy and AUC; excellent generalization and minimal overfitting. |
| Decision Tree       | Lower accuracy compared to other models; prone to variance but maintains strong precision. |
| kNN                 | Balanced performance with high recall; effective in capturing malignant cases. |
| Naive Bayes         | Moderate accuracy but very strong AUC; efficient baseline model with simple assumptions. |
| Random Forest       | Robust ensemble method; performance nearly matches Logistic Regression with strong stability. |
| XGBoost             | High recall and competitive F1 score; slightly better than Random Forest in capturing malignant cases. |

## Live App
The deployed Streamlit app is available here:  
👉 [Click to open the app](https://2025aa05840mlassignment2-xhsnudkcp86vtdjoenhytf.streamlit.app/)

## Features of the Streamlit App
- CSV dataset upload option (test data only).  
- Model selection dropdown.  
- Display of evaluation metrics (Accuracy, Precision, Recall, F1, MCC, AUC).  
- Confusion matrix visualization.  

## How to Run Locally
To run the app on your local machine:

```bash
# Clone the repository
git clone <your-repo-link>
cd project-folder

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
