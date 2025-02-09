import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import mlflow
import mlflow.sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from src.models.evaluate import evaluate_model

def train_naive_bayes(x_train_tfidf, x_test_tfidf, y_train, y_test, evaluate_model):
    """
    Train a Naive Bayes model using the provided training and testing data.
    """
    with mlflow.start_run():
        nb_model = MultinomialNB()

        param_grid = {'alpha': [0.01, 0.1, 1]}

        grid_search = GridSearchCV(nb_model, param_grid, cv=5)
        grid_search.fit(x_train_tfidf, y_train)
        best_model = grid_search.best_estimator_

        y_pred = grid_search.predict(x_test_tfidf)
        evaluate_model(y_test, y_pred, "Naive Bayes")

        nb_probabilities = best_model.predict_proba(x_test_tfidf)
        print("Naive Bayes Probabilities for each class (first 5 samples):\n", nb_probabilities[:5])
        
        # Log parameters
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)

        # Log the trained model
        mlflow.sklearn.log_model(best_model, "Naive Bayes")

        # Register the model in MLflow Model Registry
        model_uri = f"models:/Naive Bayes/latest"
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/Naive Bayes",
            name="Naive Bayes"
        )
        print("Model 'Naive Bayes' registered successfully.")

    return best_model