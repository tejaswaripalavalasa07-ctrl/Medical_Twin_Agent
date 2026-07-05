import google.generativeai as genai

# Configure API Key
import os
from dotenv import load_dotenv


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load Gemini Model
model = genai.GenerativeModel("gemini-2.5-flash")



def cardiology_agent(patient, risk, probability):

    prompt = f"""
You are a senior Cardiologist AI Agent.

Your responsibility is ONLY cardiovascular analysis.

Patient Information:

{patient}

Machine Learning Prediction:

Risk : {risk}

Probability : {probability:.2f}%

Tasks:

1. Explain the cardiovascular risk.
2. Identify important heart-related risk factors.
3. Mention which factors increase cardiovascular risk.
4. Do NOT diagnose disease.
5. Do NOT prescribe medicines.
Maximum 6 bullet points.
Maximum 120 words.
Simple English.
No paragraphs.
6. End with:
"This is an educational prototype."

"""

    response = model.generate_content(prompt)

    return response.text
patient = """
Age : 52
Blood Pressure : 150
Cholesterol : 240
Maximum Heart Rate : 145
"""

