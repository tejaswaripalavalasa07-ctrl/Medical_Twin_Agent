import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Sample patient (encoded values)
patient = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])

# Scale input
patient_scaled = scaler.transform(patient)

# Prediction
prediction = model.predict(patient_scaled)
probability = model.predict_proba(patient_scaled)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Heart Disease Risk: High")
else:
    print("Heart Disease Risk: Low")

print(f"Probability: {probability[0][1] * 100:.2f}%")