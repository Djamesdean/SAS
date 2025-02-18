"""
Module for setting up MLflow tracking with Dagshub.
"""

import os
import dagshub
import mlflow

def dags_access():
   
    mlflow.set_tracking_uri(
        "https://dagshub.com/Djamesdean/SAS-MLFLOW.mlflow"
    )
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'djamesdean'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '8e009ed06f4ac66d34599916803055c698bbb9bf'

    dagshub.init(repo_owner='Djamesdean', repo_name='SAS-MLFLOW', mlflow=True)

    # Set MLflow Tracking URI (Modify this if using a remote MLflow server)
    mlflow.set_tracking_uri("https://dagshub.com/Djamesdean/SAS-MLFLOW.mlflow")  # Change if running MLflow server elsewhere

    # Set Experiment Name
    mlflow.set_experiment("steam_reviews_sentiment")