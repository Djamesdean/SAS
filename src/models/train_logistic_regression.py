def train_logistic_regression():
  with mlflow.start_run():
    lr_model = LogisticRegression(max_iter=100)
    param_grid = {'C': [0.1, 1, 10]}

    grid_search = GridSearchCV(lr_model, param_grid, cv=5)
    grid_search.fit(X_train_tfidf, y_train)
    best_model = grid_search.best_estimator_

    y_pred = grid_search.predict(X_test_tfidf)
    evaluate_model(y_test, y_pred, "Logistic Regression")

    lr_probabilities = best_model.predict_proba(X_test_tfidf)
    print("Logistic Regression Probabilities for each class:\n", lr_probabilities[:5])

       # Log parameters
    for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)

        # Log the trained model
    mlflow.sklearn.log_model(best_model, "Logistic Regression")

        # Register the model in MLflow Model Registry
    model_uri = f"models:/{"Logistic Regression"}/latest"
    mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/{"Logistic Regression"}",
            name="Logistic Regression"
        )
    
    return best_model