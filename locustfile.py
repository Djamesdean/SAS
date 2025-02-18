import random
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)  # Wait time between requests

    # A list of example texts for sentiment analysis
    texts = [
        "I love this game!",
        "This is the worst experience ever.",
        "The graphics are amazing, but the gameplay is boring.",
        "I can't stop playing this game, it's so fun!",
        "Not what I expected, could be better.",
        "The game crashed twice in a row, very frustrating.",
        "Absolutely fantastic! Would recommend to everyone.",
        "Terrible. Waste of time and money."
    ]

    @task
    def test_sentiment(self):
        # Randomly choose a text from the list
        text_to_predict = random.choice(self.texts)

        # Send the POST request with the chosen text
        self.client.post("/predict", json={"text": text_to_predict})