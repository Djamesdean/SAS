# Steam Reviews Dataset Card

**Version:** 1.0  
**Dataset Name:** Steam Reviews Dataset  
**Author:** Anonymous 
**Source:** [Kaggle](https://www.kaggle.com/datasets/piyushagni5/sentiment-analysis-for-steam-reviews?select=train.csv)
**Date:** 20/02/2025  

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [Data Collection](#data-collection)
- [Dataset Structure](#dataset-structure)
- [Preprocessing Steps](#preprocessing-steps)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Ethical Considerations](#ethical-considerations)
- [Limitations](#limitations)

## Dataset Overview
The Steam Reviews Dataset contains user-submitted reviews from the Steam gaming platform. The dataset is labeled for sentiment classification, with user reviews categorized as **positive (1)** or **negative (0)** based on the `user_suggestion` column.

## Data Collection
- **Source:** Steam game reviews
- **Type:** Text-based dataset
- **Language:** English
- **Labels:** Binary sentiment classification (1 = Positive, 0 = Negative)
- **Size:** 17,494 reviews (after removing null values)

## Dataset Structure
Overview of the main Columns in the dataset :

| Column Name        | Description                                      |
|--------------------|--------------------------------------------------|
| `user_review`      | The actual game review written by a user        |
| `user_suggestion`  | Sentiment label: 1 (Positive), 0 (Negative)     |
| `cleaned_review`   | Preprocessed version of `user_review`           |

## Preprocessing Steps
The dataset was preprocessed using the following steps:
1. **Missing Data Handling**: Removed rows with null values.
2. **Text Cleaning**: Lowercased text and removed special characters.
3. **Tokenization**: Split text into individual words.
4. **Lemmatization**: Converted words to their base forms using WordNet Lemmatizer.
5. **Feature Extraction**: Extracted cleaned reviews for model training.

## Exploratory Data Analysis
### Sentiment Distribution
- **Positive Reviews:** 53.2%  
- **Negative Reviews:** 46.8%  

A pie chart was generated to visualize the distribution of sentiments.

### WordCloud Analysis
To better understand the frequent words in different sentiments:
1. A **WordCloud** was generated for positive reviews.
2. A **filtered WordCloud** was generated for negative reviews, removing words with neutral or positive sentiment scores.
3. Sentiment scores were calculated using **VADER SentimentIntensityAnalyzer**.

## Ethical Considerations
- **Bias:** The dataset primarily contains English reviews, which may introduce language bias.
- **Fairness:** Sentiment detection may misinterpret sarcasm or informal language.
- **Privacy:** The dataset does not include personally identifiable information.

## Limitations
- The dataset is **limited to English** and may not generalize to other languages.
- **Sarcastic or ambiguous reviews** might be misclassified.
- **Class imbalance** may affect model performance.
- The dataset does not account for **review length, time trends, or game-specific factors**.
