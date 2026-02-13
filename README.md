# ML Assignment 2

## a. Problem Statement
Implement six classification models on a dataset from UCI/Kaggle and compare their performance using multiple evaluation metrics.

## b. Dataset Description
- Dataset: Breast Cancer Wisconsin (Diagnostic)  
- Source: UCI Machine Learning Repository  
- Features: 30 numeric features describing cell nuclei characteristics  
- Target: Binary classification (Malignant = 1, Benign = 0)  
- Number of samples: 569  

## c. Models Used
The following six models were implemented:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

### Comparison Table of Evaluation Metrics

| ML Model Name      | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression|          |     |           |        |    |     |
| Decision Tree      |          |     |           |        |    |     |
| kNN                |          |     |           |        |    |     |
| Naive Bayes        |          |     |           |        |    |     |
| Random Forest      |          |     |           |        |    |     |
| XGBoost            |          |     |           |        |    |     |

*(Fill in values from your Step 2 results table)*

---

### Observations on Model Performance

| ML Model Name      | Observation about model performance |
|--------------------|--------------------------------------|
| Logistic Regression| (e.g., Stable baseline, good precision/recall balance) |
| Decision Tree      | (e.g., Overfits slightly, lower AUC compared to ensembles) |
| kNN                | (e.g., Sensitive to scaling, decent accuracy but slower with large data) |
| Naive Bayes        | (e.g., Fast, but weaker precision on this dataset) |
| Random Forest      | (e.g., Strong performance, robust against overfitting) |
| XGBoost            | (e.g., Best overall metrics, high AUC and F1) |

---

## ✅ What to Do Next
1. Run your notebook (Step 2 code).  
2. Copy the **results_df table** output.  
3. Paste the values into the **Comparison Table** above.  
4. Write **observations** for each model based on the metrics.  

---

👉 Once you fill this README, your repo will be complete up to Step 5.  
Next, we’ll move to **Step 6: Streamlit Deployment** — where you’ll use `app.py` to load models and test with CSV files.  

Would you like me to now draft the **starter `app.py` code** so you can see how the dropdown + CSV upload will work for deployment?
