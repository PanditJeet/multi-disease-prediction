import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Multi-Disease Prediction", layout="wide")

# ---------------- LOAD MODELS ----------------
diabetes_model = pickle.load(open("diabetes.pkl", "rb"))
heart_model = pickle.load(open("heart.pkl", "rb"))
stroke_model = pickle.load(open("stroke.pkl", "rb"))
cnn_model = load_model("cnn_model.h5")

# ---------------- HEADER ----------------
st.title("🏥 Multi-Disease Prediction System")
st.markdown("Predict Diabetes, Heart Disease, Stroke, and Pneumonia using ML & Deep Learning")

menu = st.sidebar.radio("Select Module",
                        ["Diabetes", "Heart Disease", "Stroke", "Pneumonia (X-ray)"])

# ---------------- DIABETES ----------------
if menu == "Diabetes":
    st.header("🩺 Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20)
        glucose = st.slider("Glucose Level", 0, 200)
        bp = st.slider("Blood Pressure", 0, 150)
        skin = st.slider("Skin Thickness", 0, 100)

    with col2:
        insulin = st.slider("Insulin Level", 0, 900)
        bmi = st.slider("BMI", 0.0, 70.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5)
        age = st.slider("Age", 1, 100)

    if st.button("Predict Diabetes"):
        data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        pred = diabetes_model.predict([data])[0]
        prob = diabetes_model.predict_proba([data])[0]

        st.subheader("Result")
        if pred == 1:
            st.error(f"High Risk ({prob[1]*100:.2f}%)")
        else:
            st.success(f"Low Risk ({prob[0]*100:.2f}%)")

# ---------------- HEART ----------------
elif menu == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 1, 100)
        sex = st.selectbox("Sex", ["Female", "Male"])
        sex = 1 if sex == "Male" else 0

        cp = st.selectbox("Chest Pain Type", ["0", "1", "2", "3"])
        cp = int(cp)

        trestbps = st.slider("Resting Blood Pressure", 80, 200)
        chol = st.slider("Cholesterol", 100, 600)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
        fbs = 1 if fbs == "Yes" else 0

        restecg = st.selectbox("Rest ECG", ["0", "1", "2"])
        restecg = int(restecg)

        thalach = st.slider("Max Heart Rate", 60, 220)

        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang = 1 if exang == "Yes" else 0

        oldpeak = st.slider("ST Depression", 0.0, 6.0)

        slope = st.selectbox("Slope", ["0", "1", "2"])
        slope = int(slope)

        ca = st.selectbox("Major Vessels", ["0", "1", "2", "3"])
        ca = int(ca)

        thal = st.selectbox("Thal", ["1", "2", "3"])
        thal = int(thal)

    if st.button("Predict Heart Disease"):
        data = [age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal]

        pred = heart_model.predict([data])[0]
        prob = heart_model.predict_proba([data])[0]

        st.subheader("Result")
        if pred == 1:
            st.error(f"High Risk ({prob[1]*100:.2f}%)")
        else:
            st.success(f"Low Risk ({prob[0]*100:.2f}%)")

# ---------------- STROKE ----------------
elif menu == "Stroke":
    st.header("🧠 Stroke Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 1, 100)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        hypertension = 1 if hypertension == "Yes" else 0

    with col2:
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        heart_disease = 1 if heart_disease == "Yes" else 0

        avg_glucose = st.slider("Average Glucose Level", 50, 300)
        bmi = st.slider("BMI", 10.0, 60.0)

    if st.button("Predict Stroke"):
        data = [age, hypertension, heart_disease, avg_glucose, bmi]

        pred = stroke_model.predict([data])[0]
        prob = stroke_model.predict_proba([data])[0]

        st.subheader("Result")
        if pred == 1:
            st.error(f"High Risk ({prob[1]*100:.2f}%)")
        else:
            st.success(f"Low Risk ({prob[0]*100:.2f}%)")

# ---------------- CNN ----------------
elif menu == "Pneumonia (X-ray)":
    st.header("🫁 Pneumonia Detection")

    uploaded_file = st.file_uploader("Upload Chest X-ray Image",
                                     type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        img = np.array(image)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.resize(img, (150, 150))
        img = img / 255.0
        img = img.reshape(1, 150, 150, 3)

        prediction = cnn_model.predict(img)[0][0]

        st.subheader("Result")
        if prediction > 0.5:
            st.error(f"Pneumonia Detected ({prediction*100:.2f}%)")
        else:
            st.success(f"Normal ({(1-prediction)*100:.2f}%)")
