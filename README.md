# 🚀 DonorAI – Predictive Fundraising & Recommendation System

An end-to-end AI system that predicts donor likelihood and generates actionable outreach strategies using Machine Learning and LLMs.

---

## 🧠 Overview

DonorAI is designed to help organizations identify potential donors and optimize outreach strategies.

Instead of just predicting outcomes, the system:

* Estimates **probability of donation**
* Generates **human-readable insights**
* Recommends **targeted fundraising actions**

---

## 🎯 Problem Statement

Organizations often struggle to:

* Identify high-value donors
* Optimize campaign targeting
* Interpret model predictions

This project solves that by combining:

* **Machine Learning (prediction)**
* **LLM (explanation + strategy)**

---

## ⚙️ System Architecture

```
User Input (Streamlit UI)
        ↓
FastAPI Backend
        ↓
ML Model (Random Forest + Pipeline)
        ↓
Threshold-based Decision
        ↓
LLM (LangChain + OpenAI)
        ↓
Final Output (Prediction + Insight + Recommendation)
```

---

## 🧪 Tech Stack

* **Backend:** FastAPI
* **Frontend:** Streamlit
* **ML:** scikit-learn
* **LLM:** LangChain + OpenAI API
* **Data Processing:** pandas, numpy

---

## 📊 Machine Learning Approach

### 🔹 Dataset

* UCI Bank Marketing Dataset (reframed for donor prediction)

### 🔹 Key Steps

* Data cleaning & preprocessing
* One-hot encoding via `ColumnTransformer`
* Feature scaling
* Train/test split with stratification

### 🔹 Model

* Random Forest Classifier

### 🔹 Improvements

* Hyperparameter tuning using GridSearchCV
* Evaluation using F1-score (not just accuracy)
* Threshold optimization for business alignment

---

## ⚠️ Important ML Decisions

### ❌ Removed Data Leakage

* Dropped `duration` feature (not available at prediction time)

### ⚖️ Handled Class Imbalance

* Used `class_weight="balanced"`

### 🎯 Threshold Tuning

* Optimized decision threshold instead of relying on default (0.5)

---

## 🧠 LLM Integration

The system uses an LLM to:

* Explain predictions in natural language
* Suggest actionable fundraising strategies

### 🔥 Key Features

* Prompt grounding (prevents hallucination)
* Rule-based constraints
* Domain-aligned recommendations

---

## 📌 Example Output

```json
{
  "probability": 0.47,
  "prediction": 0,
  "insight": "The individual shows financial constraints due to default and active loans, resulting in a low likelihood of donation.",
  "recommendation": "Use a low-pressure, personalized outreach strategy focusing on awareness rather than immediate donation requests."
}
```

---

## 🖥️ UI Preview

* Streamlit-based interactive interface
* Real-time predictions
* AI-generated insights & recommendations

---

## 🚀 How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/chiraggoyal-12/DonorAI.git
cd donor-ai
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set Environment Variables

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

### 5️⃣ Run Backend (FastAPI)

```bash
uvicorn app.app:app --reload
```

---

### 6️⃣ Run Frontend (Streamlit)

```bash
streamlit run main.py
```

---

## 📂 Project Structure

```
donor-ai/
│
├── app/               # FastAPI backend
├── model/             # ML training + prediction
├── utils/             # LLM logic
├── main.py            # Streamlit frontend
├── data/              # Dataset
├── requirements.txt
└── README.md
```

---

## 💡 Key Highlights

* End-to-end AI system (not just ML model)
* Real-world ML considerations (leakage, imbalance, thresholding)
* LLM integration for explainability
* Clean API + UI separation

---

## 🚀 Future Improvements

* Add model monitoring
* Store predictions for analytics
* A/B testing for outreach strategies
* Deploy on cloud (AWS / GCP)

---

## 👨‍💻 Author

Developed as part of an AI engineering portfolio project.

---

## ⭐ If you found this useful

Give this repo a star ⭐
