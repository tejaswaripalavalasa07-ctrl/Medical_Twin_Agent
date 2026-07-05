import streamlit as st
import pandas as pd
import joblib
from agents.cardiology_agent import cardiology_agent
from agents.diabetes_agent import diabetes_agent
from agents.safety_agent import safety_agent
from agents.judge_agent import judge_agent

# Load Model & Scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Medical Digital Twin", page_icon="❤️")

st.title("❤️ Medical Digital Twin")
st.write("Heart Disease Risk Prediction")

# ---------------- INPUTS ---------------- #

age = st.number_input("Age", 20, 100, 50)

sex = st.selectbox("Sex", ["Female", "Male"])

cp = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

bp = st.number_input("Resting Blood Pressure", 80, 220, 120)

chol = st.number_input("Cholesterol", 100, 600, 200)

fbs = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])

restecg = st.selectbox(
    "Rest ECG",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)

maxhr = st.number_input("Maximum Heart Rate", 60, 220, 150)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0)

slope = st.selectbox(
    "Slope",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

vessels = st.selectbox(
    "Colored Vessels",
    [0,1,2,3,4]
)

thal = st.selectbox(
    "Thalassemia",
    [
        "Normal",
        "Fixed Defect",
        "Reversable Defect",
        "No"
    ]
)

# ---------------- ENCODING ---------------- #

sex = 1 if sex=="Male" else 0

cp_map = {
    "Typical Angina":0,
    "Atypical Angina":1,
    "Non-anginal Pain":2,
    "Asymptomatic":3
}

fbs = 1 if fbs=="Yes" else 0

restecg_map = {
    "Normal":0,
    "ST-T Wave Abnormality":1,
    "Left Ventricular Hypertrophy":2
}

exang = 1 if exang=="Yes" else 0

slope_map = {
    "Upsloping":0,
    "Flat":1,
    "Downsloping":2
}

thal_map = {
    "Normal":0,
    "Fixed Defect":1,
    "Reversable Defect":2,
    "No":3
}

patient = pd.DataFrame([{
    "age":age,
    "sex":sex,
    "chest_pain_type":cp_map[cp],
    "resting_blood_pressure":bp,
    "cholestoral":chol,
    "fasting_blood_sugar":fbs,
    "rest_ecg":restecg_map[restecg],
    "Max_heart_rate":maxhr,
    "exercise_induced_angina":exang,
    "oldpeak":oldpeak,
    "slope":slope_map[slope],
    "vessels_colored_by_flourosopy":vessels,
    "thalassemia":thal_map[thal]
}])

# ---------------- PREDICT ---------------- #
patient_scaled = scaler.transform(patient)
probability = model.predict_proba(patient_scaled)
prediction = model.predict(patient_scaled)
risk = "High Risk" if prediction[0] == 1 else "Low Risk"
patient_info = f"""
Age : {age}
Blood Pressure : {bp}
Cholesterol : {chol}
Maximum Heart Rate : {maxhr}
"""
agent_response = cardiology_agent(
    patient_info,
    risk,
    probability[0][1] * 100
)
diabetes_response = diabetes_agent(
    patient_info,
    risk,
    probability[0][1] * 100
)
safety_response = safety_agent(
    patient_info,
    risk,
    probability[0][1] * 100
)

if st.button("Predict Risk"):

   

    


    if prediction[0] == 1:
        st.error("⚠️ High Heart Disease Risk")
    else:
        st.success("✅ Low Heart Disease Risk")

    st.write(f"Probability : {probability[0][1]*100:.2f}%")


    st.subheader("🫀 Cardiology Agent Review")
    st.info(agent_response)
    st.subheader("🩸 Diabetes Agent Review")

    st.info(diabetes_response)
    st.subheader("🛡 Safety Agent")
    st.info(safety_response)



    st.info("Educational prototype only. This system does not diagnose diseases or prescribe medication.")
    st.markdown("---")

st.subheader("🩺 Digital Twin Simulation")

new_bp = st.slider(
    "🩸 Resting Blood Pressure",
    max(80, bp - 30),
    min(220, bp + 30),
    bp
)

new_chol = st.slider(
    "🥩 Cholesterol",
    max(100, chol - 50),
    min(600, chol + 50),
    chol
)

new_hr = st.slider(
    "❤️ Maximum Heart Rate",
    max(60, maxhr - 30),
    min(220, maxhr + 30),
    maxhr
)
twin_patient = patient.copy()

twin_patient["resting_blood_pressure"] = new_bp
twin_patient["cholestoral"] = new_chol
twin_patient["Max_heart_rate"] = new_hr

twin_scaled = scaler.transform(twin_patient)

twin_prediction = model.predict(twin_scaled)

twin_probability = model.predict_proba(twin_scaled)


if st.button("Run Simulation"):


    st.write("### Original Patient")

    st.write(f"🩸 Blood Pressure : {bp}")
    st.write(f"🥩 Cholesterol : {chol}")
    st.write(f"❤️ Max Heart Rate : {maxhr}")

    st.write("### Simulated Patient")

    st.write(f"🩸 Blood Pressure : {new_bp}")
    st.write(f"🥩 Cholesterol : {new_chol}")
    st.write(f"❤️ Max Heart Rate : {new_hr}")

    st.write(f"Original Risk : {probability[0][1]*100:.2f}%")
    st.write(f"New Risk : {twin_probability[0][1]*100:.2f}%")

    if twin_prediction[0] == 1:
        st.error("High Risk")
    else:
        st.success("Low Risk")
judge_response = judge_agent(
    agent_response,          # Cardiology Agent output
    diabetes_response,
    safety_response,
    probability[0][1] * 100,      # Original Risk
    twin_probability[0][1] * 100  # Digital Twin Risk
)
st.subheader("⚖️ Judge Agent")
st.success(judge_response)