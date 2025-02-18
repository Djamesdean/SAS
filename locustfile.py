from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 5)  # Wait time between requests

    @task
    def test_sentiment(self):
        self.client.post("/predict", json={"text": "I love this game!"})