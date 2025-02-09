"""
This module contains functions to evaluate machine learning models and log their performance to MLflow.
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow

def evaluate_model(y_true, y_pred, model_name):
    """Evaluates and logs model performance to MLflow."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    print(f"\n{model_name} Evaluation:")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    mlflow.log_metric("f1_score_negative", report['0']['f1-score'])

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_positive", report['1']['precision'])
    mlflow.log_metric("recall_positive", report['1']['recall'])
    mlflow.log_metric("f1_score_positive", report['1']['f1-score'])
    mlflow.log_metric("precision_negative", report['0']['precision'])
    mlflow.log_metric("recall_negative", report['0']['recall'])
    mlflow.log_metric("f1_score_negative", report['0']['f1-score'])