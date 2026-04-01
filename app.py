import streamlit as st
import pickle

# ---------------- LOAD MODELS ----------------
diabetes_model = pickle.load(open("diabetes.pkl", "rb"))
heart_model = pickle.load(open("heart.pkl", "rb"))
stroke_model = pickle.load(open("stroke.pkl", "rb"))

# ---------------- TITLE ----------------
st.title("Multi-Disease Prediction System")

menu = st.sidebar.selectbox("Select Module",
                           ["Diabetes", "Heart Disease", "Stroke"])

# ---------------- DIABETES ----------------
if menu == "Diabetes":
    st.header("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin Level")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes"):
        data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        pred = diabetes_model.predict([data])[0]
        prob = diabetes_model.predict_proba([data])[0]

        if pred == 1:
            st.error(f"Diabetes Risk ({prob[1]*100:.2f}%)")
        else:
            st.success(f"No Diabetes Risk ({prob[0]*100:.2f}%)")

# ---------------- HEART ----------------
elif menu == "Heart Disease":
    st.header("Heart Disease Prediction")

    age = st.number_input("Age")

    sex = st.selectbox("Sex", ["Female", "Male"])
    sex = 1 if sex == "Male" else 0

    cp = st.selectbox("Chest Pain Type", ["0", "1", "2", "3"])
    cp = int(cp)

    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")

    fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0

    restecg = st.selectbox("Rest ECG", ["0", "1", "2"])
    restecg = int(restecg)

    thalach = st.number_input("Max Heart Rate")

    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0

    oldpeak = st.number_input("ST Depression")

    slope = st.selectbox("Slope", ["0", "1", "2"])
    slope = int(slope)

    ca = st.selectbox("Number of Major Vessels", ["0", "1", "2", "3"])
    ca = int(ca)

    thal = st.selectbox("Thal", ["1", "2", "3"])
    thal = int(thal)

    if st.button("Predict Heart Disease"):
        data = [age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal]

        pred = heart_model.predict([data])[0]
        prob = heart_model.predict_proba([data])[0]

        if pred == 1:
            st.error(f"Heart Disease Risk ({prob[1]*100:.2f}%)")
        else:
            st.success(f"No Heart Disease Risk ({prob[0]*100:.2f}%)")

# ---------------- STROKE ----------------
elif menu == "Stroke":
    st.header("Stroke Prediction")

    age = st.number_input("Age")

    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    hypertension = 1 if hypertension == "Yes" else 0

    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    heart_disease = 1 if heart_disease == "Yes" else 0

    avg_glucose = st.number_input("Average Glucose Level")
    bmi = st.number_input("BMI")

    if st.button("Predict Stroke"):
        data = [age, hypertension, heart_disease, avg_glucose, bmi]

        pred = stroke_model.predict([data])[0]
        prob = stroke_model.predict_proba([data])[0]

        if pred == 1:
            st.error(f"Stroke Risk ({prob[1]*100:.2f}%)")
        else:
            st.success(f"No Stroke Risk ({prob[0]*100:.2f}%)")
