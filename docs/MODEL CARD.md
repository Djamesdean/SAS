## Sentiment Analysis for Steam Reviews

**Version:** 1.0  
**Developed by:** [Chems Eddine Benteboula]  
**Date:** 20/02/2025  

## Table of Contents
- [Model Overview](#model-overview)
- [Intended Use](#intended-use)
- [Performance](#performance)
- [Ethical Considerations](#ethical-considerations)
- [Limitations](#limitations)
- [Evaluation Metrics](#evaluation-metrics)
- [Model Details](#model-details)

## Model Overview
- **Name**: Sentiment Analysis Model for Steam Reviews  
- **Description**: This model performs sentiment classification on Steam game reviews, predicting whether a review is **positive or negative**. It is trained using different machine learning algorithms, including Naive Bayes, Logistic Regression, and LSTM.
- **Algorithms**:
  - **Naive Bayes**: Multinomial Naive Bayes for text classification.
  - **Logistic Regression**: Logistic regression for binary classification.
  - **LSTM**: Long Short-Term Memory model for deep learning-based sentiment analysis.
- **Input**: Raw Steam reviews (text).
- **Output**: Sentiment (positive/negative).

## Intended Use
- This model is intended for sentiment classification tasks on video game reviews, providing insights into customer feedback.
- It can be used to analyze the general sentiment of product reviews to assist businesses in understanding customer feedback.
- It helps guide developers on future updates to provide the community with more content that keeps them engaged and willing to purchase more of their products.

## Performance
- **Naive Bayes**: Achieved **82% accuracy** with minimal training time.
- **Logistic Regression**: Similar performance with slightly better results on some data subsets (**85% accuracy**).
- **LSTM**: Trained for **10 epochs**, achieving slightly lower accuracy but consuming significantly more time and resources. It has potential for improvement with additional training.

## Ethical Considerations
- **Bias**: The model might be biased towards reviews written in English or more formal review styles, potentially misclassifying non-native or informal language reviews.
- **Fairness**: Care should be taken when using this model to avoid misinterpreting customer feedback, especially when applying it in customer service or product improvement decisions.

## Limitations
- The model is trained on **English text** and may not perform well on reviews in other languages.
- It may struggle with **sarcastic or ambiguous reviews**, where sentiment is difficult to determine.
- The **LSTM model** is more resource-intensive and requires longer processing times.

## Evaluation Metrics
- **Accuracy**, **Precision**, **Recall**, and **F1-score** for all models.
- **Confusion Matrix** to evaluate classification performance.

## Model Details
- **Training Time**:
  - Naive Bayes and Logistic Regression trained in **under 5 minutes**.
  - LSTM requires up to **50 minutes**, depending on dataset size and hardware.
- **Resources**:
  - Trained on a local machine using Python.
  - **Scikit-learn** used for Naive Bayes and Logistic Regression.
  - **TensorFlow/Keras** used for LSTM.
- **Version**: Model version **1.0** (trained in February 2025).

