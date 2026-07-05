import google.generativeai as genai

import os
from dotenv import load_dotenv


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


def diabetes_agent(patient, risk, probability):

    
  prompt = f"""
You are a Diabetes Specialist AI Agent participating in a Multi-Agent Clinical Debate.

Patient Information:
{patient}

Machine Learning Prediction:
Risk: {risk}
Probability: {probability:.2f}%

Your role is ONLY to review diabetes-related factors.

Instructions:

• Analyze only fasting blood sugar and diabetes-related information.
• Mention whether fasting blood sugar appears normal or elevated.
• Clearly state that HbA1c, glucose, insulin, and other diabetes-specific laboratory values are NOT available in this dataset.
• Explain how the lack of these values limits a complete diabetes assessment.
• Do NOT discuss cardiovascular risk (that is the Cardiology Agent's responsibility).
• Do NOT diagnose diabetes.
• Do NOT recommend medicines or treatment.
• Respond in a maximum of 3 concise bullet points.



"""
  response = model.generate_content(prompt)

  return response.text