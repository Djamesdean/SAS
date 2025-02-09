import pytest
import mlflow.sklearn


# Define the model names in MLflow
MODEL_NAMES = ["Naive Bayes", "Logistic Regression", "lstm_model"]

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_load_models(model_name):
    """
    Test loading trained models from MLflow.
    """
    try:
        # Load the latest version of the model from MLflow Model Registry
        model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        assert model is not None, f"Failed to load model: {model_name}"
        print(f"✅ Successfully loaded {model_name} model from MLflow.")
    
    except Exception as e:
        pytest.fail(f"❌ Error loading {model_name}: {e}")

if __name__ == "__main__":
    pytest.main()

    