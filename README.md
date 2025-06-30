# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning. Built during the Oasis Infobyte Data Analytics Internship, it demonstrates how models can be trained to recognize patterns and anomalies in financial datasets.

## 🧠 Problem Statement

In real-world scenarios, fraudulent transactions are rare and hard to detect. This project addresses the challenge using classification models and techniques to handle highly imbalanced data.

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Features**: 30 (PCA transformed), `Time`, `Amount`, `Class` (0: Legit, 1: Fraud)

## 🚀 Key Techniques

- Data balancing using **undersampling**
- Feature scaling with **StandardScaler**
- Binary classification using:
  - Logistic Regression
  - Random Forest Classifier
- Performance evaluation using:
  - **Confusion Matrix**
  - **Classification Report**
  - **ROC Curve**
  - **AUC Score**

## 📌 Results

| Model               | AUC Score |
|--------------------|-----------|
| Logistic Regression| 0.95+     |
| Random Forest      | 0.99+     |

> Note: These results may vary depending on train-test split random state and preprocessing steps.

## 📈 Visualizations

- Confusion matrices for both models
- ROC curve comparison

## 🧪 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python fraud_detection.py
