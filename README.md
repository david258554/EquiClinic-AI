# EquiClinic AI

Clinical Bias Auditing System — CECS 451 Term Project

## Overview

EquiClinic AI audits clinical decision support systems (CDSS) for demographic bias. It generates synthetic patient profiles using GPT-4o-mini, classifies risk using a decision tree, and detects bias using demographic parity.

## AI Techniques Used

| Technique | Chapter |
| LLM API (GPT-4o-mini) |
| Decision Tree Classifier |
| Entropy and Information Gain |
| Demographic Parity Bias Detection |

## Installation

pip install openai python-dotenv

## Configuration

Create a .env file with your OpenAI API key:

OPENAI_API_KEY=your-key-here

## Usage

python equiclinic.py "chest pain"
python equiclinic.py "back pain"
python equiclinic.py "headache"

## Example Output

Symptom: CHEST PAIN
Male patients: 10
Female patients: 10
Male High Risk Rate: 60.0%
Female High Risk Rate: 0.0%
Bias Score: 0.600
Threshold: 10%
[ALERT] BIAS DETECTED - Demographic disparity exceeds threshold.

## Output Files

audit_report.json - Full audit results in JSON format

