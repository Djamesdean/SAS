import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer



def preprocess_data(data):
    
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

    data['cleaned_review'] = data['user_review'].apply(preprocess_text)

    # Extract features and target variables
    x = data['cleaned_review']
    y = data['user_suggestion']
    # Split dataset into training and test sets (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
    x_train_tfidf = tfidf.fit_transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    return x_train_tfidf, x_test_tfidf, y_train, y_test
