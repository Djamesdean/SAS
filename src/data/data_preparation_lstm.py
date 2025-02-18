import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer





def preprocess_data_lstm(data):
    
    data.dropna(inplace=True)
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

