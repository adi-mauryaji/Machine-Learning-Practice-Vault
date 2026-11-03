from pathlib import Path

import joblib
import pandas as pd
import streamlit as st 

# Load the saved files frim the same folder as app.py

folder = Path(__file__).parent
model = joblib.load(folder / "model.pkl")
scaler = joblib.load(folder / "scalar.pkl")

st.title("Diabetes Prediction App")
st.write("For my ML Practice , not Medical Advice. Test accuracy is 70.8%.")

# Label, minimum and maximum values for each feature
pregnancies = st.number_input("Pregnancies", 0, 20, 3)
glucose = st.number_input("Glucose", 0, 300, 117)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 72)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 23)
insulin = st.number_input("Insulin", 0, 900, 30)
bmi = st.number_input("BMI", 0.0, 70.0, 32.0)
pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.372, format="%.3f")
age = st.number_input("Age", 1, 120, 29)

if st.button("Predict"):
    data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }])

    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("Prediction: The person is likely to have diabetes.")
    else:
        st.success("Prediction: The person is unlikely to have diabetes.")