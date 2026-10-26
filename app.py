import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

# Load model
model = joblib.load("titanic_rf_model.joblib")
le_sex = joblib.load("le_sex.joblib")
le_embarked = joblib.load("le_embarked.joblib")
feature_names = joblib.load("feature_names.joblib")

st.title("🚢 Titanic Survival Predictor")
st.write("Predict if a passenger would survive the Titanic disaster")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["female", "male"])
age = st.slider("Age", 0.5, 80.0, 30.0)
sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 30.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

if st.button("Predict"):
    sex_enc = le_sex.transform([sex])[0]
    emb_enc = le_embarked.transform([embarked])[0]
    
    input_data = pd.DataFrame([[pclass, sex_enc, age, sibsp, parch, fare, emb_enc]], 
                               columns=feature_names)
    
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    if pred == 1:
        st.success(f"Survived! Probability: {prob*100:.1f}%")
    else:
        st.error(f"Did Not Survive. Survival Probability: {prob*100:.1f}%")