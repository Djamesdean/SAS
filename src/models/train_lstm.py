import tensorflow as tf  
import matplotlib.pyplot as plt  
import seaborn as sns 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.model_selection import KFold  
from sklearn.model_selection import cross_val_score
import mlflow

def train_lstm(X_train,X_test,y_train,y_test):
    
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
        model.add(Embedding(input_dim=5000, output_dim=100, input_length=100 ,weights=[embedding_matrix], 
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

     # Save final model to MLflow
        mlflow.keras.log_model(final_model, "LSTM_Model")

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
