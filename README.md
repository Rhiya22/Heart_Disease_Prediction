# Heart_Disease_Prediction

This project applies machine learning techniques to predict the likelihood of heart disease in patients based on clinical data. The goal is to demonstrate end-to-end ML workflow from data cleaning and feature engineering to model training, evaluation, and interpretation.

# Project Overview

Built a predictive model using Python (pandas, scikit-learn, matplotlib, seaborn).

Preprocessed the dataset (handling missing values, scaling features, encoding categories).

Explored multiple ML models (Logistic Regression, Random Forest, Gradient Boosting).

Evaluated models using accuracy, precision, recall, F1-score, and ROC-AUC.

Applied SHAP values to improve explainability of predictions.

# Dataset

Source: UCI Heart Disease dataset (Cleveland) or equivalent clinical dataset.

Features include:

Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Induced Angina, ST Depression, Slope, Major Vessels, Thalassemia, etc.

Target: Presence (1) or absence (0) of heart disease.

# Tech Stack

Languages: Python

Libraries: pandas, NumPy, scikit-learn, matplotlib, seaborn, shap

Environment: Jupyter Notebook

# How to Run
# Clone the repository
git clone https://github.com/Rhiya22/heart_disease_prediction.git
cd heart-disease-prediction

# Run the Jupyter Notebook
jupyter notebook Heart_Disease_Prediction.ipynb


# Results

Best model achieved ~85–90% accuracy (depending on train/test split).

Random Forest and Gradient Boosting performed best.

SHAP analysis revealed important features such as chest pain type, max heart rate, ST depression, and age.

# Example Output

ROC-AUC Curve

Confusion Matrix

Feature Importance (SHAP values)

# Future Improvements

Hyperparameter tuning with GridSearch/RandomSearch.

Ensemble methods for better generalization.

Deployment as a Flask or Streamlit web app.

# Author

Rhiya Raman

https://www.linkedin.com/in/rhiya-raman/

rhiyaraman2001@gmail.com
