import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def data_visualize(data):
    balance = data['user_suggestion'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(balance, labels=balance.index, autopct='%1.1f%%', startangle=90)
    plt.title('Sentiment Distribution as Percentage')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

def plot_wordclouds(df):
    analyzer = SentimentIntensityAnalyzer()
    def get_sentiment_score(text):
        
        scores = analyzer.polarity_scores(text)
        return scores['compound']
    
    # Combine all reviews for each sentiment
    positive_reviews = " ".join(df[df['user_suggestion'] == 1]['cleaned_review'])
    negative_reviews = " ".join(df[df['user_suggestion'] == 0]['cleaned_review'])

    positive_word_scores = {word: get_sentiment_score(word) for word in positive_reviews.split()}
    negative_word_scores = {word: get_sentiment_score(word) for word in negative_reviews.split()}

    # Create WordCloud objects
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


