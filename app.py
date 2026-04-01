import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# ---------------- LOAD MODELS ----------------
diabetes_model = pickle.load(open("diabetes.pkl", "rb"))
heart_model = pickle.load(open("heart.pkl", "rb"))
stroke_model = pickle.load(open("stroke.pkl", "rb"))
cnn_model = load_model("cnn_model.h5")

# ---------------- TITLE ----------------
st.title("Multi-Disease Prediction System")

menu = st.sidebar.selectbox("Select Module", 
                           ["Diabetes", "Heart Disease", "Stroke", "Pneumonia (X-ray)"])

# ---------------- DIABETES ----------------
if menu == "Diabetes":
    st.header("Diabetes Prediction")

    inputs = [st.number_input(f"Feature {i}") for i in range(8)]

    if st.button("Predict"):
        pred = diabetes_model.predict([inputs])[0]
        prob = diabetes_model.predict_proba([inputs])[0]

        if pred == 1:
            st.error(f"Diabetes Risk ({prob[1]*100:.2f}%)")
        else:
            st.success(f"No Diabetes Risk ({prob[0]*100:.2f}%)")

# ---------------- HEART ----------------
elif menu == "Heart Disease":
    st.header("Heart Disease Prediction")

    inputs = [st.number_input(f"Feature {i}") for i in range(13)]

    if st.button("Predict"):
        pred = heart_model.predict([inputs])[0]
        prob = heart_model.predict_proba([inputs])[0]

        if pred == 1:
            st.error(f"Heart Disease Risk ({prob[1]*100:.2f}%)")
        else:
            st.success(f"No Heart Disease Risk ({prob[0]*100:.2f}%)")

# ---------------- STROKE ----------------
elif menu == "Stroke":
    st.header("Stroke Prediction")

    inputs = [st.number_input(f"Feature {i}") for i in range(len(stroke_model.feature_names_in_))]

    if st.button("Predict"):
        pred = stroke_model.predict([inputs])[0]
        prob = stroke_model.predict_proba([inputs])[0]

        if pred == 1:
            st.error(f"Stroke Risk ({prob[1]*100:.2f}%)")
        else:
            st.success(f"No Stroke Risk ({prob[0]*100:.2f}%)")

# ---------------- CNN (X-RAY) ----------------
elif menu == "Pneumonia (X-ray)":
    st.header("Pneumonia Detection from X-ray")

    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = np.array(image)
        img = cv2.resize(img, (150,150))
        img = img / 255.0
        img = img.reshape(1,150,150,3)

        prediction = cnn_model.predict(img)[0][0]

        if prediction > 0.5:
            st.error(f"Pneumonia Detected ({prediction*100:.2f}%)")
        else:
            st.success(f"Normal ({(1-prediction)*100:.2f}%)")