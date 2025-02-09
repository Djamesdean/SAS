import pytest
import mlflow.sklearn


# Define the model names in MLflow
#add lstm after logistic regression in the model names
MODEL_NAMES = ["Naive Bayes", "Logistic Regression"]

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_load_models(model_name):
    
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        assert model is not None, f"Failed to load model: {model_name}"
        print(f"✅ Successfully loaded {model_name} model from MLflow.")
    
    except Exception as e:
        pytest.fail(f"❌ Error loading {model_name}: {e}")



if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"], plugins=[CaptureOutput(f)])
    with open("test_results.txt", "w") as f:
    pytest.main(["-v", "--tb=short"], stdout=f)

    