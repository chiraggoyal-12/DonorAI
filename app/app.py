from fastapi import FastAPI
from pydantic import BaseModel

from model.predict import predict

app = FastAPI(title = "Donor Prediction API")

class DonorRequest(BaseModel):
  age: int
  job: str
  marital: str
  education: str
  default: str
  balance: int
  housing: str
  loan: str
  contact: str
  day: int
  month: str
  campaign: int
  pdays: int
  previous: int
  poutcome: str


@app.post("/")
def home():
  return {"message": "Welcome to the Donor Prediction API!"}

@app.post("/predict")
def predict_api(request: DonorRequest):
  data = request.dict()
  result = predict(data)
  return result