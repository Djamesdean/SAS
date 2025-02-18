import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware
import time
from pydantic import BaseModel
from models.model_loader import (
    naive_bayes,
    logistic_regression,
    
)
from data.data_preparation import preprocess_text
import numpy as np

app = FastAPI()

# Serve static files (e.g., icons, favicon)
app.mount("/statics", StaticFiles(directory="statics"), name="statics")

REQUEST_COUNT = Counter("request_count", "Total requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])
ERROR_COUNT = Counter("error_count", "Total errors", ["status_code", "endpoint"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction processing time", ["model"])
MODEL_USAGE = Counter("model_usage_count", "Total model usage", ["model"])
POSITIVE_PREDICTIONS = Counter("positive_predictions", "Total positive predictions")
NEGATIVE_PREDICTIONS = Counter("negative_predictions", "Total negative predictions")

# Middleware for monitoring
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        method = request.method
        endpoint = request.url.path
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()

        start_time = time.time()
        response = await call_next(request)
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)

        return response

app.add_middleware(MetricsMiddleware)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type="text/plain")



# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

# Favicon endpoint
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("statics/favicon.ico")

class TextRequest(BaseModel):
    text: str
    model: str = "logistic_regression"  # Default model

@app.post("/predict")
async def predict_sentiment(request: TextRequest):
    try:
        # Clean input text
        cleaned_text = preprocess_text(request.text)
        
        # Choose model
        if request.model == "naive_bayes":
            prediction = naive_bayes.predict(cleaned_text)[0]
        elif request.model == "logistic_regression":
            prediction = logistic_regression.predict(cleaned_text)[0]
        elif request.model == "LSTM":
            # Add LSTM-specific preprocessing
            prediction = lstm_model.predict(cleaned_text)[0].item()
        else:
            raise HTTPException(status_code=400, detail="Invalid model specified")
        
        return {
            "text": request.text,
            "model": request.model,
            "sentiment": "positive" if prediction == 1 else "negative",
            "confidence": float(np.max(prediction)) if request.model == "lstm" else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))