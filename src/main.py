# Import libraries for data processing, modeling, and evaluation
import mlflow
import pandas as pd
import re  
import matplotlib.pyplot as plt  
import seaborn as sns 
import numpy as np  
from sklearn.model_selection import train_test_split, GridSearchCV, KFold  
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.model_selection import cross_val_score


from src.data.load_data import Preprocess_data, load_data
from src.features.mlflow import dags_access
from src.models import train_logistic_regression, train_lstm, train_naive_bayes
from src.models.evaluate import evaluate_model
from src.visualization.visualize_data import data_visualize, plot_wordclouds

data = pd.read_csv('data/raw/Steam.csv')
data = pd.read_csv('data/raw/Steam.csv')
print(data.shape)
print(f"Dataset shape: {data.shape}")

def main():
    load_data()
    Preprocess_data(data)
    data_visualize()
    plot_wordclouds(data)
    dags_access()    
    evaluate_model()
    train_naive_bayes()
    train_logistic_regression()
    train_lstm()

