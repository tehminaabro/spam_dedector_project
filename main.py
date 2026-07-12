

from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

svm_model = None
tfidf_vectorizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global svm_model, tfidf_vectorizer
    
    # Aap ke computer ka bilkul sahi absolute path:
    model_path = r"C:\Users\AA.Y TRADEERS\Desktop\spam_dedector_project\spam_model_svm.joblib"
    vectorizer_path = r"C:\Users\AA.Y TRADEERS\Desktop\spam_dedector_project\tfidf_vectorizer.joblib"
    
    # Ab files bina kisi error ke load hongi
    
    svm_model = joblib.load("spam_model_svm.joblib") 
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib") 
    yield


app = FastAPI(lifespan=lifespan)

class EmailInput(BaseModel):
    email_text: str

@app.get("/")
def home():
    return {"message": "Spam Detector API is Running Successfully!"}

@app.post("/predict")
def predict_email(data: EmailInput):
    vectorized_data = tfidf_vectorizer.transform([data.email_text])
    prediction = svm_model.predict(vectorized_data)
    result = "Spam" if prediction == 1 else "Ham (Safe)"
    return {
        "email_received": data.email_text,
        "prediction": result
    }
