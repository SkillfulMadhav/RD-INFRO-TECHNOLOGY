import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("AI-Powered Medical Diagnosis System")
st.warning("For educational purposes only. Not a medical diagnosis tool.")

age = st.slider("Age", 20, 80)
sex = st.selectbox("Sex", [0, 1])
cp = st.slider("Chest Pain Type", 0, 3)
trestbps = st.slider("Resting Blood Pressure", 80, 200)
chol = st.slider("Cholesterol", 100, 400)
thalach = st.slider("Maximum Heart Rate", 70, 210)

if st.button("Predict"):
    input_data = np.array([[
    age,          # age
    sex,          # sex
    cp,           # chest pain
    trestbps,     # resting BP
    chol,         # cholesterol
    0,            # fasting blood sugar (normal)
    1,            # resting ECG (normal)
    thalach,      # max heart rate
    0,            # exercise induced angina (no)
    0.0,          # ST depression
    1,            # slope (normal)
    0,            # number of vessels (0 blocked)
    2             # thal (normal)
]])

    prob = model.predict_proba(input_data)[0][1]

    if prob > 0.6:
        st.error("Possible Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
