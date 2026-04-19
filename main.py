import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Donor AI", layout="centered")

st.title("🎯 Donor Prediction System")
st.markdown("Predict donor behavior and get AI-powered recommendations")

# -----------------------------
# Input Form
# -----------------------------
with st.form("donor_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    job = st.selectbox("Job", ["management", "technician", "entrepreneur", "blue-collar", "services", "admin.", "student", "unemployed", "retired", "self-employed", "unknown"])
    marital = st.selectbox("Marital Status", ["single", "married", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Has Credit Default?", ["yes", "no"])
    balance = st.number_input("Account Balance", value=1000)
    housing = st.selectbox("Housing Loan", ["yes", "no"])
    loan = st.selectbox("Personal Loan", ["yes", "no"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
    day = st.slider("Last Contact Day", 1, 31, 10)
    month = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    campaign = st.number_input("Campaign Contacts", value=1)
    pdays = st.number_input("Days Since Last Contact (-1 = never)", value=-1)
    previous = st.number_input("Previous Contacts", value=0)
    poutcome = st.selectbox("Previous Outcome", ["unknown", "failure", "success", "other"])

    submit = st.form_submit_button("Predict")

# -----------------------------
# Prediction
# -----------------------------
if submit:
    payload = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            st.subheader("📊 Prediction Result")

            st.metric("Probability of Donation", f"{result['probability']:.2f}")
            st.metric("Prediction", "Likely to Donate" if result["prediction"] == 1 else "Unlikely to Donate")

            st.subheader("🧠 AI Insight")
            st.write(result.get("insight", "No insight available"))

            st.subheader("🎯 Recommendation")
            st.write(result.get("recommendation", "No recommendation available"))

        else:
            st.error("API error. Check backend.")

    except Exception as e:
        st.error("Failed to connect to API. Is FastAPI running?")