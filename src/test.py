import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from models.model_loader import (
    naive_bayes,
    logistic_regression,
    
)
from data.data_preparation import preprocess_text
import numpy as np

text = "this game is amazing"
text2 = preprocess_text(text)
prediction = naive_bayes.predict(text2)
print("Predicted Sentiment:", prediction)
