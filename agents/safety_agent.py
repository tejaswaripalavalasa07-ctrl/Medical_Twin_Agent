import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def safety_agent(patient, risk, probability):

    prompt = f"""
You are a Medical Safety Reviewer AI Agent.

Patient Information:
{patient}

ML Prediction:
Risk: {risk}
Probability: {probability:.2f}%

Your responsibility is ONLY to review the safety of the AI response.

Instructions:

• Check whether the prediction is presented as an educational estimate.
• Ensure no diagnosis is made.
• Ensure no medicines or dosage recommendations are given.
• Mention that additional clinical information may be required.
• Remind users to consult a qualified healthcare professional.
• Maximum 4 bullet points.
• End with:
"Educational prototype only."

"""

    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
    return response.text 