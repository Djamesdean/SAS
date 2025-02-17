import re
import nltk
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib



nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
vectorizer = joblib.load('/Users/merkava/Documents/School/Software engineering/SAS/src/data/tfidf_vectorizer.pkl')

def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Tokenize text
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            token_text=  ' '.join(tokens)

            text_tfidf = vectorizer.transform([token_text])
            return text_tfidf