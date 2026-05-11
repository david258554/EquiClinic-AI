"""
EquiClinic AI - Clinical Bias Auditing System
CECS 451 Term Project
David Rivera
Due Date 5/12/2026

AI Techniques Implemented:
1. LLM API Usage (GPT-4o-mini) 
2. Decision Tree Classifier 
3. Entropy & Information Gain 
4. Demographic Parity Bias Detection

"""

import random 
import math 
import json
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# load api key from .env file
load_dotenv()
client = OpenAI()

# LLM patient generation 

def generate_patients_with_llm(symptom, num_patients=20):
    """
    Uses GPT-4o-mini API to generate synthetic patient profiles.
    API key is loaded from .env file (not in code).
    """
    prompt = f"""
    Generate {num_patients} synthetic patient profiles with the symptom: {symptom}.
    Include realistic variation in age (20-80), BMI (18-40), and gender (Male/Female).
    Also include symptom_severity (1-10 scale, higher = more severe).
    Return ONLY a JSON array, no explanation. Example format:
    [{{"symptom": "{symptom}", "age": 55, "bmi": 28, "gender": "Male", "symptom_severity": 7}}]
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7  
    )
    
    raw = response.choices[0].message.content.strip()


    # Clean up potential markdown formatting
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    
    patients = json.loads(raw)
    return patients


# decision tree

def entropy(p):
    "Entropy: B(q) = -(q log2 q + (1-q) log2 (1-q))"
    if p == 0 or p ==1:
        return 0
    return -(p * math.log2(p) + (1-p) * math.log2(1-p))

def decision_tree_clasify(age, bmi, gender, symptom_severity):
    "Manual decision tree for clinical risk assessment"
    if age >50:
        if bmi > 30:
            return "High Risk"
        else:
            return "Medium Risk"
    else:
        if gender == "Male":
            if symptom_severity > 7:
                return "High Risk"
            else:
                return "Medium Risk"
        else:
            if symptom_severity > 8:
                return "Medium Risk"
            else:
                return "Low Risk"