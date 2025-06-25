# credit-card-fraud-detection-ml
# Credit Card Fraud Detection with Machine Learning

A comprehensive machine learning project for detecting credit card fraud using various algorithms and data balancing techniques on the popular Kaggle Credit Card Fraud Detection dataset.

## 🔍 Overview
This project implements multiple machine learning models to detect fraudulent credit card transactions using the **Kaggle Credit Card Fraud Detection Dataset**. The dataset is highly imbalanced (typical in fraud detection scenarios), so we employ random undersampling and SMOTE (Synthetic Minority Oversampling Technique) to create balanced training sets for fair model evaluation.

## 📊 Dataset
**Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features**: 30 features (V1-V28 PCA transformed, Time, Amount)
- **Target**: Binary classification - 0 (Legitimate) vs 1 (Fraud)
- **Challenge**: Highly imbalanced dataset (~99.8% legitimate transactions)
- **Size**: 284,807 transactions with 492 fraudulent cases

## 🛠️ Models Implemented
- **Neural Networks (NN)** - Deep learning approach with dropout regularization
- **Neural Networks + SMOTE** - Enhanced with synthetic data generation
- **Multi-Layer Perceptron (MLP)** - Simplified neural network architecture
- **Random Forest** - Ensemble method with decision trees
- **Logistic Regression** - Linear classification baseline
- **K-Nearest Neighbors (KNN)** - Instance-based learning
- **Support Vector Machine (SVM)** - Kernel-based classification

## 🎯 Key Features
- **Data Balancing**: Random undersampling and SMOTE implementation
- **Feature Standardization**: StandardScaler for Time and Amount features
- **Comprehensive Evaluation**: Precision, Recall, F1-score, and Confusion Matrix
- **Visualization**: Class distribution plots and correlation heatmaps
- **Model Comparison**: Side-by-side performance analysis of all algorithms
