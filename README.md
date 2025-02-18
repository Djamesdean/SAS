Sentiment Analysis for Steam Reviews
==============================

working on sentiment analysis project for video games reviews in the steam platform .
we are using various comments from different videos games in steam to try and predict the overall sentiment of the text.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---
# INCEPTION 

To kick off this project, we began by defining the scope and objectives using the Machine Learning Canvas. This helped us align with the stakeholder's requirements and outline the key components of the project, including data collection, model development, and evaluation metrics.

We created as small overall report about the project [Report](./docs/SA-Report.pdf) and a graphic representation of the canvas [Canva](./docs/Canva.pdf).

To ensure a standardized and organized project structure, we used Cookiecutter (data science), a tool for generating project templates. 

---
# REPRODUCIBILITY 

 To achieve a well orgnized project that could be used and shared by other devolopers, we implemented  :

## Code Versioning
We used Git for version control, tracking all changes to the codebase and ensuring that modifications are well-documented and reversible.
The repository is hosted on GitHub, where collaborators can access the latest version of the code and contribute to the project.


## Experiment Tracking
To manage the end-to-end lifecycle of the machine learning project, we used MLflow. MLflow enables consistent tracking of experiments, recording parameters, metrics, and results systematically. This promotes transparency and reproducibility.

## DagsHub Integration
We used DagsHub as a unified platform for version control, data versioning, and experiment tracking. DagsHub integrates seamlessly with Git, DVC, and MLflow, enhancing collaboration and reproducibility.

[LINK TO DAGSHUB | MLFLOW](https://dagshub.com/Djamesdean/SAS)

---
# Quality Assurance 
Quality Assurance (QA) ensures that your models, data, and results are reliable, reproducible, and robust. It helps avoid errors, ensures consistency, and improves the credibility of your work.
Here’s a breakdown of the approach taken in this project : 

## pylint :
Pylint is a Python static code analysis tool that checks your code for errors, style issues, and potential bugs. It follows the PEP 8 coding standard and helps improve code readability, maintainability, and quality.
**Why is Pylint Useful for Your Sentiment Analysis Project?**
- Detects Errors & Bugs 🛑

Catches undefined variables, unused imports, and incorrect function calls.
Helps prevent runtime errors in data preprocessing, model training, and inference.

- Improves Code Readability 📖
Enforces PEP 8 guidelines (consistent spacing, naming conventions).
Makes the code easier to read and maintain, especially in collaborative projects.

- Ensures Best Practices ✅
Highlights inefficient loops and redundant operations in ML model training.
Suggests refactoring to optimize performance (e.g., list comprehensions).

- Enhances Reproducibility 🔄
Standardizes the code style, making it easier to debug and extend in the future.

[Pylint-Report](./reports/pylint_report.json)

## pytest :
Pytest is a Python testing framework used for writing and running unit tests. It helps verify that your code functions correctly and remains error-free after modifications.
**Why is Pytest Useful for Your Sentiment Analysis Project?`**

- Automates Testing ⚙️
Easily run tests on different parts of your code with one command.
Helps catch errors early, avoiding issues in model training and deployment.

- Tests ML Model Performance 📊
Verify that Models like in our case Naive Bayes, Logistic Regression, and LSTM load and predict correctly.
Ensure that your TF-IDF vectorization, label encoding, and evaluation functions produce expected results.

- Speeds Up Debugging 🔍
Provides detailed error messages when something goes wrong.
Allows for parameterized testing to check multiple inputs at once.

### Test Functions :

#### test_preprocess_data
Tests if the preprocess_data function correctly processes the data.
- x_train_tfidf and x_test_tfidf are sparse matrices (expected for TF-IDF).
- y_train and y_test are non-empty.
- The transformed feature space has exactly 5000 features (matching TF-IDF max_features).

#### test naive_bayes
Tests if the train_naive_bayes function correctly trains and predicts.
Makes predictions on the test set.
Assertions:
- Predictions (y_pred) match the length of y_test.
- The accuracy score is between 0 and 1.

#### test Logistic_regression
Tests if the train_logistic_regression function correctly trains and predicts.
Makes predictions on the test set.
- Predictions (y_pred) match the length of y_test.
- The accuracy score is between 0 and 1.

#### test Lstm
Tests if the train_lstm function correctly trains and predicts.
- Uses a small subset of 20 samples for fast testing.
- prediction matches test set size.
- Accuracy is between 0 and 1.
- Predictions contain only 0 and 1.

[Test File](./src/tests/test_pipeline.py)
[Test Results](./src/tests/pytest_results.txt)

## Great Expectations 
is a data validation and profiling tool that helps ensure data quality. It allows you to define, test, and document "expectations" (rules) about your data.
since we are working on sentiment analysis we applied some simple expectation on our data to make sure it is ready for training.
- ensure the user_review column has no missing values.
- ensure user_suggestion column only has values of 1 and 0.

[Great Expectation file](./src/tests/test_data_validation.py)
[GreatExpectation Results1](./src/tests/validation_result_user_review.json)
[GreatExpectation Results2](./src/tests/validation_result_user_suggestion.json)

---

# API
APIs (Application Programming Interfaces) are sets of tools that allow software applications to communicate with each other. APIs define the methods and data formats that applications can use to request and exchange information. 
**Sentiment Analysis APIs:** designed to analyze the sentiment of text, returning whether the text is positive, negative. These APIs use machine learning models and predefined algorithms to predict sentiment based on the input text.

## FastAPI 
provides a simple API for sentiment analysis using various machine learning models, such as Naive Bayes, Logistic Regression, and LSTM. Here's a breakdown of its functionality:

- **TextRequest for Model:** Defines a data model for the request body that should include
Defines a data model for the request body that should include:
    - text: The input text to analyze.
    - model: A string specifying which model to use for prediction ("logistic_regression", "naive_bayes", or "LSTM"). Default is "logistic_regression".

**Prediction Endpoint** The core endpoint that performs sentiment analysis. This is a POST request that accepts a TextRequest object and returns sentiment predictions.
- Flow:
Preprocessing: The input text is first cleaned using the preprocess_text function .
- Model Selection:
Based on the model specified in the request, the appropriate model (naive_bayes, logistic_regression, or LSTM) is used to predict the sentiment.
If LSTM is chosen, additional preprocessing specific to LSTM is applied (this part isn't fully shown in the code, but you'd likely have LSTM preprocessing steps).
- Prediction: 
The model returns a prediction, where:
1 represents positive sentiment.
0 represents negative sentiment.
Response:
The endpoint responds with a JSON object containing:
The original text.
The model used.
The sentiment (positive or negative).

[API File](./src/main.py)

---

# MODEL AND DATASET CARD

## Model Card:
A model card is a structured document that provides details about an ML model, including its purpose, performance, limitations, and ethical considerations. It helps users understand how the model was built, how it performs, and the best ways to use it.

**Why We Used a Model Card?**
- Improves transparency by documenting model performance and behavior.
- Helps users decide whether the model is suitable for their use case.
- Highlights potential biases and ethical considerations.
- Provides technical details for reproducibility.
[MODEL CARD](./docs/MODEL%20CARD.md)

## Dataset Card:
A dataset card is a structured documentation format for datasets, describing their source, composition, intended use, and limitations. It ensures responsible dataset usage and reproducibility.

**Why We Used a Dataset Card?**
Documents dataset collection, cleaning, and preprocessing steps.
Helps users understand the dataset’s potential biases and ethical implications.
Ensures reproducibility for experiments using the dataset.
Facilitates better data management and sharing.

[DARASET CARD](./docs/DATASET%20CARD.md)

---











<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
