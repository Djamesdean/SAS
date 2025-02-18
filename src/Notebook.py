# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import re 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.naive_bayes import MultinomialNB  
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split, GridSearchCV, KFold 
import seaborn as sns 
import joblib
import mlflow 
import mlflow.sklearn
import dagshub
import os

import tensorflow as tf  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences






# %%
data = pd.read_csv('/Users/merkava/Documents/School/Software engineering/SAS/data/raw/Steam.csv')
data.dropna(inplace=True)

print(data.shape)
print(f"Dataset shape: {data.shape}")



# %%
balance = data['user_suggestion'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(balance, labels=balance.index, autopct='%1.1f%%', startangle=90)
plt.title('Sentiment Distribution as Percentage')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# %%
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize text
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)


# %%
data['cleaned_review'] = data['user_review'].apply(preprocess_text)

# Extract features and target variables
X = data['cleaned_review']
y = data['user_suggestion']  
# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%

analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
  scores = analyzer.polarity_scores(text)
  return scores['compound']

def plot_wordclouds(df):

  # Combine all reviews for each sentiment
  positive_reviews = " ".join(df[df['user_suggestion'] == 1]['cleaned_review'])
  negative_reviews = " ".join(df[df['user_suggestion'] == 0]['cleaned_review'])

  positive_word_scores = {word: get_sentiment_score(word) for word in positive_reviews.split()}
  negative_word_scores = {word: get_sentiment_score(word) for word in negative_reviews.split()}

  # Create WordCloud objects
  #positive_word_counts = Counter(positive_reviews.split())
  negative_word_counts = Counter(negative_reviews.split())

  # Create WordCloud objects with frequency and sentiment score weighting
  wordcloud_positive = WordCloud(width=800, height=400, background_color='white', 
                                  stopwords=STOPWORDS).generate_from_frequencies(positive_word_scores)

  # Filter out words with positive sentiment scores from the negative word cloud
  negative_word_scores_filtered = {word: score * negative_word_counts[word] 
                                   for word, score in negative_word_scores.items() if score < 0}

  wordcloud_negative = WordCloud(width=800, height=400, background_color='white', 
                                  stopwords=STOPWORDS).generate_from_frequencies(negative_word_scores_filtered)
  # Display the generated images
  plt.figure(figsize=(20, 10))
  plt.subplot(1, 2, 1)
  plt.imshow(wordcloud_positive, interpolation='bilinear')
  plt.title('Positive Reviews')
  plt.axis("off")

  plt.subplot(1, 2, 2)
  plt.imshow(wordcloud_negative, interpolation='bilinear')
  plt.title('Negative Reviews')
  plt.axis("off")

  plt.show()

plot_wordclouds(data)


# %%
os.environ['MLFLOW_TRACKING_USERNAME'] = 'djamesdean'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '8e009ed06f4ac66d34599916803055c698bbb9bf'

dagshub.init(repo_owner='Djamesdean', repo_name='SAS', mlflow=True)

# Set MLflow Tracking URI (Modify this if using a remote MLflow server)
mlflow.set_tracking_uri("https://dagshub.com/Djamesdean/SAS.mlflow")  

# Set Experiment Name
mlflow.set_experiment("Steam")

# %%
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
    plt.show()

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_positive", report['1']['precision'])
    mlflow.log_metric("recall_positive", report['1']['recall'])
    mlflow.log_metric("f1_score_positive", report['1']['f1-score'])
    mlflow.log_metric("precision_negative", report['0']['precision'])
    mlflow.log_metric("recall_negative", report['0']['recall'])
    mlflow.log_metric("f1_score_negative", report['0']['f1-score'])

# %%
def train_naive_bayes(X_train, X_test, y_train, y_test):
   tfidf = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
   X_train_tfidf = tfidf.fit_transform(X_train)
   X_test_tfidf = tfidf.transform(X_test)
   with mlflow.start_run(): 
    nb_model = MultinomialNB()

    param_grid = {'alpha': [0.01, 0.1, 1]}

    grid_search = GridSearchCV(nb_model, param_grid, cv=5)
    grid_search.fit(X_train_tfidf, y_train)
    best_model = grid_search.best_estimator_

    y_pred = grid_search.predict(X_test_tfidf)
    evaluate_model(y_test, y_pred, "Naive Bayes")

    nb_probabilities = best_model.predict_proba(X_test_tfidf)
    print("Naive Bayes Probabilities for each class (first 5 samples):\n", nb_probabilities[:5])
        # Log parameters
        
    for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)

        # Log the trained model
    mlflow.sklearn.log_model(best_model, "Naive Bayes")

        # Register the model in MLflow Model Registry
    model_uri = f"models:/{"Naive Bayes"}/latest"
    mlflow.register_model(
       model_uri=f"runs:/{mlflow.active_run().info.run_id}/{"Naive Bayes"}",
       name="Naive Bayes"
        )
    print(f"Model {"Naive Bayes"} registered successfully.")

    return  best_model

# %%
def train_logistic_regression(X_train,X_test,y_train,y_test):
  
  tfidf = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
  X_train_tfidf = tfidf.fit_transform(X_train)
  X_test_tfidf = tfidf.transform(X_test)

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

# %%
def train_lstm():
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=200)

    embedding_index = {}
    with open('/Users/merkava/Documents/School/Machine Learning/glove.6B.100d.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    embedding_matrix = np.zeros((5000, 100))
    for word, i in tokenizer.word_index.items():
        if i < 5000:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
    
    # Define LSTM model creation function
    def create_lstm_model():
        model = Sequential()
       
        model.add(Embedding(input_dim=5000, output_dim=100, input_length=100 , weights=[embedding_matrix],
                            trainable=False))
        model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))
        model.add(Dense(64, activation='relu'))  # Additional dense layer
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # K-Fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    with mlflow.start_run():  # Start MLflow tracking
        mlflow.log_param("model", "LSTM")
        mlflow.log_param("epochs", 10)
        mlflow.log_param("batch_size", 64)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_seq)):
            print(f"\nFold {fold + 1}")

            X_train_fold, X_val_fold = X_train_seq[train_idx], X_train_seq[val_idx]
            y_train_array = np.array(y_train)  # Convert Series to NumPy array
            y_train_fold, y_val_fold = y_train_array[train_idx], y_train_array[val_idx]

            # Create a new model for each fold
            lstm_model = create_lstm_model()

            # Train the model
            lstm_model.fit(X_train_fold, y_train_fold, epochs=5, batch_size=64, 
                        validation_data=(X_val_fold, y_val_fold), verbose=1)

            # Evaluate on validation set
            y_val_pred = (lstm_model.predict(X_val_fold) > 0.5).astype("int32")
            accuracy = accuracy_score(y_val_fold, y_val_pred)
            fold_metrics.append(accuracy)
            print(f"Validation Accuracy for Fold {fold + 1}: {accuracy:.4f}")

        avg_accuracy = np.mean(fold_metrics)
        mlflow.log_metric("avg_validation_accuracy", avg_accuracy)

        # Display Average Accuracy Across Folds
        print("\nAverage Validation Accuracy Across Folds:", np.mean(fold_metrics))

        # Final Training on Entire Training Data
        final_model = create_lstm_model()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        final_model.fit(X_train_seq, y_train, epochs=10, batch_size=64, validation_data=(X_val_fold, y_val_fold), 
            verbose=1, callbacks=[early_stopping])
        
            # ✅ Save model to MLflow

        model_uri = f"models:/{"LSTM"}/latest"
        mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/{"LSTM"}",
            name="LSTM"
        )
      
        # Final Predictions and Evaluation
        y_test_pred = (final_model.predict(X_test_seq) > 0.5).astype("int32")

        accuracy = accuracy_score(y_test, y_test_pred)
        mlflow.log_metric("test_accuracy", accuracy)

        report = classification_report(y_test, y_test_pred, target_names=["negative", "positive"], output_dict=True)
        mlflow.log_metric("precision_positive", report["positive"]["precision"])
        mlflow.log_metric("recall_positive", report["positive"]["recall"])
        mlflow.log_metric("f1_score_positive", report["positive"]["f1-score"])

        # Display Evaluation Metrics
        print("\nLSTM Model Evaluation:")
        accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_test_pred, target_names=["negative", "positive"]))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
    plt.title("LSTM Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.show()
    return final_model


# %%
run_id = "3eed1bf3aade44708a189d0c7bca9f96"  # Replace with actual Run ID
model_uri = f"runs:/{run_id}/lstm_model"

# Register the model
mlflow.register_model(model_uri, "lstm_model")

# %%
nb_model = train_naive_bayes(X_train, X_test, y_train, y_test)

# %%
lr_model = train_logistic_regression(X_train, X_test, y_train, y_test)

# %%
lstm_model = train_lstm()


