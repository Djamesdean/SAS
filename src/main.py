from features.mlflow import dags_access
from models.evaluate import evaluate_model
from models.train_logistic_regression import train_logistic_regression
from models.train_naive_bayes import train_naive_bayes
from visualization.visualize_data import data_visualize, plot_wordclouds
from data.load_data import preprocess_data
import pandas as pd

data = pd.read_csv('data/raw/Steam.csv')

print(data.shape)
print(f"Dataset shape: {data.shape}")

def main():
    
    x_train_tfidf, x_test_tfidf, y_train, y_test= preprocess_data(data)

    train_logistic_regression(x_train_tfidf, y_train, x_test_tfidf, y_test, evaluate_model)
    train_naive_bayes(x_train_tfidf, x_test_tfidf, y_train, y_test, evaluate_model)

    
    

if __name__ == "__main__":
    main()
    
   


