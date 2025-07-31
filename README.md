# Telecom Customer Churn Prediction

This project predicts whether a telecom customer will churn (leave the service) or stay, based on their account and usage data. It uses Machine Learning to identify high-risk customers so that businesses can take preventive actions.

---

## Project Overview

* Objective: Predict customer churn for a telecom company
* Dataset: Telco Customer Churn Dataset from Kaggle
* Key Features:

  * Customer demographics (gender, senior citizen, dependents)
  * Account information (tenure, contract type, payment method)
  * Service usage (internet, streaming, online security)
  * Billing information (monthly and total charges)
* Best Model Used: Random Forest Classifier

---

## Project Workflow

1. Data Preprocessing

   * Converted categorical variables to numerical
   * Handled missing values in TotalCharges
   * Applied feature scaling

2. Model Training

   * Compared multiple models (Logistic Regression, Random Forest, XGBoost)
   * Addressed class imbalance using SMOTE

3. Deployment

   * Created an interactive web application using Streamlit
   * Allows users to input customer data and predict churn probability

---

## How to Run the Project

1. Clone the repository from GitHub

2. Install the dependencies listed in `requirements.txt`

3. Run the Streamlit application using the command:

   ```
   streamlit run app.py
   ```

4. The app will open in a browser at `http://localhost:8501`

---

## Project Structure

* app.py – Streamlit application file
* random\_forest\_model.pkl – Trained machine learning model
* scaler.pkl – Preprocessing scaler for numeric features
* requirements.txt – Project dependencies
* README.md – Project documentation


