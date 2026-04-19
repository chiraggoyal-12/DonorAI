import joblib
import pandas as pd

from utils.llm import generate_insight

model = joblib.load("model/model.pkl")

THRESHOLD = 0.55

def predict(data: dict):
  df = pd.DataFrame([data])
  prob = model.predict_proba(df)[0][1]
  prediction = int(prob >= THRESHOLD)

  return {
    "probability": prob,
    "prediction": prediction
  }

sample = {
  "age": 35,
  "job": "management",
  "marital": "single",
  "education": "tertiary",
  "default": "no",
  "balance": 5000,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day": 5,
  "month": "may",
  "campaign": 2,
  "pdays": -1,
  "previous": 0,
  "poutcome": "unknown"
}

print(predict(sample))

def predict(data: dict):
  df = pd.DataFrame([data])

  prob = model.predict_proba(df)[0][1]
  prediction = int(prob >= THRESHOLD)

  llm_output = generate_insight(data, prob)

  return {
    "probability": float(prob),
    "prediction": prediction,
    "insight": llm_output["insight"],
    "recommendation": llm_output["recommendation"]
  }