# test_sentiment_analysis.py
import pytest
import pandas as pd
import sys
import os
from unittest.mock import patch
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.data.load_data import preprocess_data
from src.data.data_preparation_lstm import preprocess_data_lstm
from src.models.train_logistic_regression import train_logistic_regression
from src.models.train_naive_bayes import train_naive_bayes
from src.models.train_lstm import train_lstm
from src.models.evaluate import evaluate_model
import scipy.sparse

@pytest.fixture
def data():
    data = pd.read_csv('/Users/merkava/Documents/School/Software engineering/SAS/data/raw/Steam.csv')
    data.dropna(inplace=True)
    return data

@pytest.fixture
def preprocessed_data(data):
    return preprocess_data(data)

@pytest.fixture
def preprocessed_data_lstm(data):
    return preprocess_data_lstm(data)


def test_preprocess_data(preprocessed_data):
    # Sample data
    x_train_tfidf, x_test_tfidf, y_train, y_test = preprocessed_data
    # Check if output is a sparse matrix
    assert scipy.sparse.issparse(x_train_tfidf)
    assert scipy.sparse.issparse(x_test_tfidf)
    
    # Check if target variables are properly split
    assert len(y_train) > 0
    assert len(y_test) > 0
    
    # Check if transformed data has expected dimensions
    assert x_train_tfidf.shape[1] == 5000  # Ensure TF-IDF features are correct
    assert x_test_tfidf.shape[1] == 5000



def test_naive_bayes(preprocessed_data):
    x_train_tfidf, x_test_tfidf, y_train, y_test = preprocessed_data
    
    with patch("mlflow.start_run"), \
         patch("mlflow.log_param"), \
         patch("mlflow.sklearn.log_model"), \
         patch("mlflow.register_model"):
     model = train_naive_bayes(x_train_tfidf, x_test_tfidf, y_train, y_test)
    y_pred = model.predict(x_test_tfidf)
    assert len(y_pred) == len(y_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert 0 <= accuracy <= 1

def test_logistic_regression(preprocessed_data):
    x_train_tfidf, x_test_tfidf, y_train, y_test = preprocessed_data

    with patch("mlflow.start_run"), \
         patch("mlflow.log_param"), \
         patch("mlflow.sklearn.log_model"), \
         patch("mlflow.register_model"):
        
        model = train_logistic_regression(x_train_tfidf, y_train, x_test_tfidf, y_test)
    
    # Predictions
    y_pred = model.predict(x_test_tfidf)
    
    # Assertions
    assert len(y_pred) == len(y_test)  # Ensure predictions match test size
    accuracy = accuracy_score(y_test, y_pred)
    assert 0 <= accuracy <= 1  # Ensure valid accuracy range
    assert set(y_pred).issubset({0, 1})

'''def test_lstm(preprocessed_data_lstm):
    x_train, x_test, y_train, y_test = preprocessed_data_lstm
    x_train_subset, y_train_subset = x_train[:20], y_train[:20]
    x_test_subset, y_test_subset = x_test[:20], y_test[:20]

    with patch("mlflow.start_run"), \
         patch("mlflow.log_param"), \
         patch("mlflow.keras.log_model"), \
         patch("mlflow.log_metric"):
        
        model = train_lstm(x_train_subset, x_test_subset, y_train_subset, y_test_subset)

    # Predictions
    y_pred = (model.predict(pad_sequences(x_test, maxlen=200)) > 0.5).astype("int32")
    
    # Assertions
    assert len(y_pred) == len(y_test_subset)  # Ensure predictions match test size
    accuracy = accuracy_score(y_test_subset, y_pred)
    assert 0 <= accuracy <= 1  # Ensure valid accuracy range
    assert set(y_pred.flatten()).issubset({0, 1})'''