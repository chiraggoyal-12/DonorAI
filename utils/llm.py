import os
from typing import Dict

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
  model = "gpt-4o-mini",
  temperature = 0.4
)

prompt_template = ChatPromptTemplate.from_template("""
  You are a fundraising AI assistant.
                             
  Your task:
  1. Explain the prediction using ONLY the provided data
  2. Suggest a specific outreach strategy

  STRICT RULES:
  - Do NOT assume any missing information
  - Use ONLY the given fields
  - Financial stress is TRUE only if default = "yes" OR loan = "yes"
  - If default = "no" AND loan = "no" → DO NOT mention financial difficulty
  - You MUST follow the probability interpretation exactly

  Probability Interpretation (MANDATORY):
  - If probability < 0.5 → MUST say LOW likelihood
  - If 0.5 ≤ probability < 0.7 → MUST say MODERATE likelihood
  - If probability ≥ 0.7 → MUST say HIGH likelihood   
                                                        - Recommendation MUST be related to fundraising strategy
  - Do NOT suggest financial counseling, workshops, or external support services
  - Focus only on donation engagement tactics                  -Prefer direct campaign actions (email, call, message) over general activities                                         
                                                 User Details:
  Age: {age}
  Job: {job}
  Marital: {marital}
  Education: {education}
  Default: {default}
  Balance: {balance}
  Housing Loan: {housing}
  Personal Loan: {loan}
  Contact Type: {contact}
  Last Contact Day: {day}
  Month: {month}
  Campaign contacts: {campaign}
  Days since last contact: {pdays}
  Previous contacts: {previous}
  Previous outcome: {poutcome}
                                                   
  Prediction Probability: {probability}
  
  Format:
  Insight: <fact based explanation here>
  Recommendation: <clear action>                         
""")

def generate_insight(data: Dict, probability: float) -> Dict:
  try:
    chain = prompt_template | llm
    response = chain.invoke({
      "age": data.get("age"),
      "job": data.get("job"),
      "marital": data.get("marital"),
      "education": data.get("education"),
      "default": data.get("default"),
      "balance": data.get("balance"),
      "housing": data.get("housing"),
      "loan": data.get("loan"),
      "contact": data.get("contact"),
      "day": data.get("day"),
      "month": data.get("month"),
      "campaign": data.get("campaign"),
      "pdays": data.get("pdays"),
      "previous": data.get("previous"),
      "poutcome": data.get("poutcome"),
      "probability": round(probability,2)
    })

    content = response.content.strip()

    insight = ""
    recommendation = ""

    if "Recommendation" in content:
      parts = content.split("Recommendation:")
      insight = parts[0].replace("Insight:", "").strip()
      recommendation = parts[1].strip()
    else:
      insight = content.strip()
      recommendation = "No recommendation provided"
      
    return {
      "insight": insight,
      "recommendation": recommendation
    }
  
  except Exception as e:
    return{
      "insight": "LLM insight unavailable",
      "recommendation": "use probability score for outreach decision"
    }
