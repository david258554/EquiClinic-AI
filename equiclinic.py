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

import math 
import json
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


BIAS_THRESHOLD = 0.10

# LLM patient generation 

def generate_patients_with_llm(symptom, num_patients=20):
    """
    Uses GPT-4o-mini API to generate synthetic patient profiles.
    API key is loaded from .env file (not in code).
    """
    prompt = f"""
Generate {num_patients} synthetic patient profiles for patients presenting with: {symptom.upper()}.

IMPORTANT: The symptom is "{symptom}". Generate patients that are REALISTIC for this specific symptom.

Follow these guidelines for {symptom}:
- Chest pain: age 45-80, severity 6-10, higher BMI
- Back pain: age 30-65, severity 4-8, normal BMI
- Headache: age 20-50, severity 3-7, any BMI
- Shortness of breath: age 50-80, severity 5-9, higher BMI
- General symptoms: age 25-70, severity 2-8, normal BMI

For each patient, include:
- age (integer, follow symptom-specific range)
- bmi (float, 18-40)
- gender ("Male" or "Female" - balanced)
- symptom_severity (integer 1-10, follow symptom-specific range)

Return ONLY a valid JSON array, no markdown, no explanation. Example format:
[{{"symptom": "{symptom}", "age": 55, "bmi": 28.5, "gender": "Male", "symptom_severity": 7}}]
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

def information_gain(patients, split_attr, split_val, target_label="High Risk"):
    """
    Computes information gain for a binary split on split_attr at split_val.
    Gain(A) = B(p/p+n) - weighted entropy of subsets.
    Validates that our decision tree splits are meaningful (Chapter 19).
    """
    n_total = len(patients)
    if n_total == 0:
        return 0.0
    
    p_total = sum(1 for p in patients if p.get("_risk") == target_label)
    parent_entropy = entropy(p_total / n_total)

    left  = [p for p in patients if p.get(split_attr, 0) <= split_val]
    right = [p for p in patients if p.get(split_attr, 0) >  split_val]

    weighted = 0.0 
    for subset in [left, right]:
        if not subset:
            continue 
        p_sub = sum(1 for p in subset if p.get("_risk") == target_label)
        weighted += (len(subset) / n_total) * entropy(p_sub / len(subset))

    return round(parent_entropy - weighted, 4)


def decision_tree_classify(age, bmi, gender, symptom_severity):
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
            

# bias detection
def demographic_parity_bias_detection(results_by_gender, threshold=0.10):
    """Demographic parity bias detection"""
    male_high_risk = results_by_gender.get("Male", {}).get("High Risk", 0)
    male_total = results_by_gender.get("Male", {}).get("Total", 1)
    female_high_risk = results_by_gender.get("Female", {}).get("High Risk", 0)
    female_total = results_by_gender.get("Female", {}).get("Total", 1)

    male_rate = male_high_risk / male_total if male_total > 0 else 0
    female_rate = female_high_risk / female_total if female_total > 0 else 0

    disparity = abs(male_rate - female_rate)

    return {
        "male_high_risk_rate": male_rate,
        "female_high_risk_rate": female_rate,
        "bias_score": disparity,
        "threshold": threshold,
        "bias_detected": disparity > threshold
    }


# main 

def main():

    if len(sys.argv) < 2:
        print("Usage: python equiclinic.py <symptom>")
        sys.exit(1)

    symptom = sys.argv[1]
    print("  EQUICLINIC AI — CLINICAL BIAS AUDITING SYSTEM")


    
    print(f"\n[INPUT]  Symptom:  {symptom.upper()}")

    # step 1: generate patients 
    
    print("[STEP 1] Generating synthetic patients via LLM")
    
    patients = generate_patients_with_llm(symptom, num_patients=20)
    print(f"[OK] {len(patients)} patient profiles generated\n")
 
    # ── Step 2: Classify with decision tree ───────────────────
    
    print("[STEP 2] Classifying risk via Decision Tree")
    
 
    results_by_gender = {
        "Male":   {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0, "Total": 0},
        "Female": {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0, "Total": 0}
    }
 
    for p in patients:
        age      = p.get("age", 50)
        bmi      = p.get("bmi", 25)
        gender   = p.get("gender", "Unknown")
        severity = p.get("symptom_severity", 5)
 
        risk = decision_tree_classify(age, bmi, gender, severity)
        p["_risk"] = risk  # tag for information gain computation
 
        if gender in results_by_gender:
            results_by_gender[gender][risk]  += 1
            results_by_gender[gender]["Total"] += 1
 
    print(f"   Male patients:   {results_by_gender['Male']['Total']}")
    print(f"   Female patients: {results_by_gender['Female']['Total']}\n")
 
    # ── Step 2b: Information gain on age split ────────────────
    ig_age = information_gain(patients, "age", 50)
    ig_sev = information_gain(patients, "symptom_severity", 7)
    print(f"   Information Gain — age > 50:            {ig_age}")
    print(f"   Information Gain — symptom_severity > 7:{ig_sev}\n")
 
    # ── Step 3: Bias detection ────────────────────────────────

    print("[STEP 3] Detecting bias (Demographic Parity)")
    
 
    bias = demographic_parity_bias_detection(results_by_gender)
 
    print(f"   Male High Risk Rate:   {bias['male_high_risk_rate']:.1%}")
    print(f"   Female High Risk Rate: {bias['female_high_risk_rate']:.1%}")
    print(f"   Bias Score:            {bias['bias_score']:.3f}")
    print(f"   Threshold:             {bias['threshold']:.0%}")
 
    if bias["bias_detected"]:
        print("\n   [ALERT] BIAS DETECTED — Demographic disparity exceeds threshold.")
        print("           Model should not be deployed without review.")
    else:
        print("\n   [OK] No significant bias detected.")
 
    # ── Save report ───────────────────────────────────────────
    report = {
        "audit_id":       datetime.now().isoformat(),
        "symptom":        symptom,
        "llm_model":      "gpt-4o-mini",
        "ai_techniques": {
            "patient_generation": "LLM API — GPT-4o-mini",
            "classification":     "Decision Tree with Entropy",
            "bias_metric":        "Demographic Parity"
        },
        "information_gain": {
            "age_split_50":            ig_age,
            "severity_split_7":        ig_sev
        },
        "results": {
            "by_gender":    results_by_gender,
            "bias_analysis": bias
        }
    }
 
    with open("audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
 
    
    print("[FILE] Report saved to: audit_report.json")

 
 
if __name__ == "__main__":
    main()