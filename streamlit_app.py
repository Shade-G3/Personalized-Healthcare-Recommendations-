
import streamlit as st
import joblib
import numpy as np

model = joblib.load('blood_donation_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Blood Donation Prediction")

rec = st.number_input("Recency", 0)
freq = st.number_input("Frequency", 0)
mon = st.number_input("Monetary", 0)
time = st.number_input("Time", 0)

if st.button("Predict"):
    arr = np.array([[rec, freq, mon, time]])
    scaled = scaler.transform(arr)
    pred = model.predict(scaled)[0]
    st.write("Likely Donor" if pred == 1 else "Unlikely Donor")
