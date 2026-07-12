
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import joblib

svm_model = None
tfidf_vectorizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global svm_model, tfidf_vectorizer
    # Render (Linux server) par direct root files load hongi
    svm_model = joblib.load("spam_model_svm.joblib") 
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib") 
    yield

app = FastAPI(lifespan=lifespan)

class EmailInput(BaseModel):
    email_text: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Spam Detector</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #f0f2f5; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .container { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); text-align: center; width: 450px; box-sizing: border-box; }
            h2 { color: #1a73e8; margin-top: 0; font-size: 26px; }
            p { color: #666; font-size: 14px; margin-bottom: 20px; }
            textarea { width: 100%; height: 120px; padding: 12px; border: 2px solid #dadce0; border-radius: 8px; resize: none; box-sizing: border-box; font-size: 15px; outline: none; transition: border-color 0.2s; }
            textarea:focus { border-color: #1a73e8; }
            button { background-color: #1a73e8; color: white; border: none; padding: 12px 20px; margin-top: 15px; border-radius: 8px; cursor: pointer; font-size: 16px; font-weight: bold; width: 100%; transition: background-color 0.2s; }
            button:hover { background-color: #1557b0; }
            #result { margin-top: 25px; font-weight: bold; font-size: 20px; min-height: 30px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>AI Spam Detector</h2>
            <p>Enter your email text below to analyze if it's safe or spam.</p>
            <textarea id="messageInput" placeholder="Paste your text or email content here..."></textarea>
            <button onclick="checkSpam()">Analyze Text</button>
            <div id="result"></div>
        </div>

        <script>
            async function checkSpam() {
                const text = document.getElementById('messageInput').value;
                const resultDiv = document.getElementById('result');
                if(!text.trim()) { resultDiv.innerText = "⚠️ Please enter some text!"; resultDiv.style.color = "#e53935"; return; }
                
                resultDiv.innerText = "⏳ Analyzing text...";
                resultDiv.style.color = "#555";

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ email_text: text })
                    });
                    const data = await response.json();
                    
                    if(data.prediction === "Spam") {
                        resultDiv.innerText = "🚨 Warning: This is SPAM!";
                        resultDiv.style.color = "#d32f2f";
                    } else {
                        resultDiv.innerText = "✅ Clean: This is HAM (Safe)!";
                        resultDiv.style.color = "#2e7d32";
                    }
                } catch (error) {
                    resultDiv.innerText = "❌ Error connecting to server.";
                    resultDiv.style.color = "#d32f2f";
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
def predict_email(data: EmailInput):
    vectorized_data = tfidf_vectorizer.transform([data.email_text])
    prediction = svm_model.predict(vectorized_data)
    result = "Spam" if prediction == 1 else "Ham (Safe)"
    return {
        "email_received": data.email_text,
        "prediction": result
    }

