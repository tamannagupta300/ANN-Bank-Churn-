import streamlit as st
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Bank Churn Prediction (ANN - Sigmoid)")

credit = st.number_input("Credit Score")
age = st.number_input("Age")
balance = st.number_input("Balance")
salary = st.number_input("Estimated Salary")

if st.button("Predict"):
    data = scaler.transform([[credit, age, balance, salary]])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Customer will leave ❌")
    else:
        st.success("Customer will stay ✅")
