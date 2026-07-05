import pandas as pd

# Load Dataset
df = pd.read_csv("data/HeartDiseaseTrain-Test.csv")

# Remove Duplicates
df.drop_duplicates(inplace=True)

print("Duplicates:", df.duplicated().sum())

print("\nSex:")
print(df["sex"].unique())

print("\nChest Pain Type:")
print(df["chest_pain_type"].unique())

print("\nFasting Blood Sugar:")
print(df["fasting_blood_sugar"].unique())

print("\nRest ECG:")
print(df["rest_ecg"].unique())

print("\nExercise Induced Angina:")
print(df["exercise_induced_angina"].unique())

print("\nSlope:")
print(df["slope"].unique())

print("\nVessels:")
print(df["vessels_colored_by_flourosopy"].unique())

print("\nThalassemia:")
print(df["thalassemia"].unique())
# ---------------- Manual Encoding ---------------- #

# Sex
df["sex"] = df["sex"].map({
    "Female": 0,
    "Male": 1
})

# Chest Pain Type
df["chest_pain_type"] = df["chest_pain_type"].map({
    "Typical angina": 0,
    "Atypical angina": 1,
    "Non-anginal pain": 2,
    "Asymptomatic": 3
})

# Fasting Blood Sugar
df["fasting_blood_sugar"] = df["fasting_blood_sugar"].map({
    "Lower than 120 mg/ml": 0,
    "Greater than 120 mg/ml": 1
})

# Rest ECG
df["rest_ecg"] = df["rest_ecg"].map({
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2
})

# Exercise Induced Angina
df["exercise_induced_angina"] = df["exercise_induced_angina"].map({
    "No": 0,
    "Yes": 1
})

# Slope
df["slope"] = df["slope"].map({
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
})

# Vessels
df["vessels_colored_by_flourosopy"] = df["vessels_colored_by_flourosopy"].map({
    "Zero": 0,
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4
})

# Thalassemia
df["thalassemia"] = df["thalassemia"].map({
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversable Defect": 2,
    "No": 3
})

# Save preprocessed dataset
df.to_csv("data/heart_preprocessed.csv", index=False)

print("Preprocessing Completed!")
print(df.head())
print(df.isnull().sum())


