                                                 Insurance Charges Prediction (ANN Model)

A machine learning web application built using TensorFlow (ANN) and Streamlit to predict medical insurance charges based on user demographic and health-related information.

This project demonstrates an end-to-end ML workflow — from data preprocessing and model training to deployment with an interactive, professional UI.

🚀 Live Features

📊 Predicts insurance charges instantly

🧠 Uses a pretrained Artificial Neural Network (ANN)

🎨 Clean, modern Streamlit dashboard UI

⚡ Fast inference with cached resources

🔁 Input handling identical to training pipeline

🧠 Model Overview

Algorithm: Artificial Neural Network (ANN)

Framework: TensorFlow / Keras

Problem Type: Regression

Target Variable: Insurance Charges

Input Features
Feature	Description
Age	Age of the individual
BMI	Body Mass Index
Children	Number of dependents
Sex	Male / Female
Smoker	Smoking status
Region	Residential region

Categorical features are one-hot encoded, and numerical features are scaled using a saved scaler.

🛠️ Tech Stack

Python 3.10+

TensorFlow / Keras

Pandas & NumPy

Scikit-learn (Scaler)

Streamlit (Frontend)

Pickle (Model artifacts)
Prediction Workflow

User enters details via UI

Inputs converted to training format

Feature scaling applied using saved scaler

ANN model predicts insurance charges

Result displayed instantly
