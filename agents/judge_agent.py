import google.generativeai as genai

import os
from dotenv import load_dotenv


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


def judge_agent(cardio_report,
                diabetes_report,
                safety_report,
                original_risk,
                simulated_risk):

    prompt = f"""
You are the Judge AI Agent in a Multi-Agent Clinical Debate.

Cardiology Agent Report:
{cardio_report}

Diabetes Agent Report:
{diabetes_report}

Safety Agent Report:
{safety_report}

Original ML Prediction Risk:
{original_risk:.2f}%

Digital Twin Simulation Risk:
{simulated_risk:.2f}%

Tasks:

• Summarize the findings from all three specialist agents.
• Compare the Original ML Risk with the Digital Twin Simulation Risk.
• Mention whether the simulated changes increased or decreased the predicted risk.
• Mention any limitations or missing information.
• Do NOT change the ML prediction.
• Do NOT diagnose disease.
• Do NOT prescribe medicines.
• Give the final report in a maximum of 6 concise bullet points.
• End with:
"Final educational summary only."

"""

    response = model.generate_content(prompt)

    return response.text