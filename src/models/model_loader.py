import mlflow 
import os
import dagshub
from mlflow import pyfunc


def load_fromdags(model_name):
  
    # Initialize DAGsHub and MLflow integration
    dagshub.init(repo_owner="Djamesdean", repo_name="SAS", mlflow=True)
    # Construct the model URI
    model_uri = f"models:/{model_name}/latest"

    # Load the model
    try:
        model = pyfunc.load_model(model_uri)
        print(f"Successfully loaded model '{model_name}'" )
        return model
    except Exception as e:
        print(f"Failed to load model '{model_name}'")
        raise
    return model


naive_bayes = load_fromdags("Naive Bayes")
logistic_regression = load_fromdags("Logistic Regression")

