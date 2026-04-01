import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Multi-Disease System", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.big-title {
    font-size:30px;
    font-weight:bold;
    color:#2E86C1;
}
.card {
    padding:20px;
    border-radius:10px;
    background-color:#f0f2f6;
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
diabetes_model = pickle.load(open("diabetes.pkl", "rb"))
heart_model = pickle.load(open("heart.pkl", "rb"))
stroke_model = pickle.load(open("stroke.pkl", "rb"))
cnn_model = load_model("cnn_model.h5")

# ---------------- HEADER ----------------
st.markdown('<p class="big-title">🏥 Multi-Disease Prediction Dashboard</p>', unsafe_allow_html=True)
st.write("AI-powered system for predicting multiple diseases")

menu = st.sidebar.radio("Select Module",
                        ["Diabetes", "Heart Disease", "Stroke", "Pneumonia"])

# ---------------- FUNCTION FOR RISK BAR ----------------
def show_result(pred, prob):
    risk = prob[1] * 100
    st.progress(int(risk))

    if pred == 1:
        st.error(f"High Risk ({risk:.2f}%)")
    else:
        st.success(f"Low Risk ({100-risk:.2f}%)")

# ---------------- DIABETES ----------------
if menu == "Diabetes":
    st.header("🩺 Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Pregnancies", 0, 15)
        glucose = st.slider("Glucose", 0, 200)
        bp = st.slider("Blood Pressure", 0, 150)
        skin = st.slider("Skin Thickness", 0, 100)

    with col2:
        insulin = st.slider("Insulin", 0, 900)
        bmi = st.slider("BMI", 0.0, 70.0)
        dpf = st.slider("DPF", 0.0, 2.5)
        age = st.slider("Age", 1, 100)

    if st.button("Predict"):
        data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        pred = diabetes_model.predict([data])[0]
        prob = diabetes_model.predict_proba([data])[0]

        st.subheader("Result")
        show_result(pred, prob)

# ---------------- HEART ----------------
elif menu == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 1, 100)
        sex = st.selectbox("Sex", ["Female", "Male"])
        sex = 1 if sex == "Male" else 0

        cp = int(st.selectbox("Chest Pain", ["0", "1", "2", "3"]))
        trestbps = st.slider("Blood Pressure", 80, 200)
        chol = st.slider("Cholesterol", 100, 600)

    with col2:
        fbs = st.selectbox("FBS > 120", ["No", "Yes"])
        fbs = 1 if fbs == "Yes" else 0

        restecg = int(st.selectbox("Rest ECG", ["0", "1", "2"]))
        thalach = st.slider("Max Heart Rate", 60, 220)

        exang = st.selectbox("Angina", ["No", "Yes"])
        exang = 1 if exang == "Yes" else 0

        oldpeak = st.slider("ST Depression", 0.0, 6.0)
        slope = int(st.selectbox("Slope", ["0", "1", "2"]))
        ca = int(st.selectbox("Vessels", ["0", "1", "2", "3"]))
        thal = int(st.selectbox("Thal", ["1", "2", "3"]))

    if st.button("Predict"):
        data = [age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal]

        pred = heart_model.predict([data])[0]
        prob = heart_model.predict_proba([data])[0]

        st.subheader("Result")
        show_result(pred, prob)

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

        avg_glucose = st.slider("Glucose", 50, 300)
        bmi = st.slider("BMI", 10.0, 60.0)

    if st.button("Predict"):
        data = [age, hypertension, heart_disease, avg_glucose, bmi]

        pred = stroke_model.predict([data])[0]
        prob = stroke_model.predict_proba([data])[0]

        st.subheader("Result")
        show_result(pred, prob)

# ---------------- CNN ----------------
elif menu == "Pneumonia":
    st.header("🫁 Pneumonia Detection")

    uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

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
