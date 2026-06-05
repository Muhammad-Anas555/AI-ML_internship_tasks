# Clinical Data Pipeline: Heart Disease Prediction

This repository contains a modular machine learning pipeline designed to process clinical patient data and predict binary heart disease risk.

## 🎯 Task Objective
The objective of this task is to construct an end-to-end classification pipeline that ingests patient medical records, checks data integrity, trains binary classifiers, and evaluates diagnostic predictions using standard medical evaluation metrics.

## 📊 Dataset Used
* **Dataset Name:** Heart Disease UCI Dataset
* **Target Variable:** Binary Classification (`0` = Low Risk / Healthy, `1` = High Risk / Diagnosed)
* **Key Patient Attributes:** Clinical metrics including age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol levels (chol), fasting blood sugar (fbs), maximum heart rate achieved (thalach), and exercise-induced angina (exang).

## Project Architecture
The codebase is structured into modules to separate data handling from machine learning operations:
heart_disease_project
   data_pipeline.py   # Loads data and applies median/mode imputation for missing entries.
   modeling.py        # Trains classifiers, runs predictions, and calculates ROC-AUC.
   analytics.py       # Extracts statistical weights and features driving the predictions.
   main.py            # The root execution script orchestrating the full pipeline.

## Models Applied
Logistic Regression: Used as a baseline classification model to map data probabilities across a linear boundary.

Decision Tree Classifier: Used to capture non-linear, step-based clinical diagnostic rules by segmenting patient features at optimized splits.

## Key Results and Findings
Clinical Stratification: Due to the critical nature of medical classifications, dataset splitting utilizes stratification (stratify=y). This ensures that the proportion of healthy to diagnosed patients remains identical across both training and testing sets.

Diagnostic Thresholds: Model performance is validated beyond simple accuracy scores by tracking the Confusion Matrix (to monitor dangerous False Negatives) and the ROC-AUC Curve (to measure the model's ability to distinguish between healthy and high-risk patients).

Feature Importance Insights: Programmatic feature weight extraction ranks specific patient metrics—such as chest pain type, maximum heart rate achieved, and age—as the strongest mathematical predictors for determining heart disease risk.