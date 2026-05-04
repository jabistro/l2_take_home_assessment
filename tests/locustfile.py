"""
Load test for the claim detection API using Locust.

Run against a live server:
    locust -f tests/locustfile.py --host=http://localhost:8000

Headless run (CI / quick benchmark):
    locust -f tests/locustfile.py --host=http://localhost:8000 \
        --headless -u 50 -r 10 --run-time 60s \
        --csv=locust_results/results
"""

import random

from locust import HttpUser, between, task

CLAIM_SENTENCES = [
    "The Eiffel Tower stands 330 metres tall.",
    "Unemployment dropped to 3.7% last quarter.",
    "The moon is approximately 384,400 km from Earth.",
    "Carbon dioxide levels hit 420 ppm in 2023.",
    "The United States has 50 states.",
]

NON_CLAIM_SENTENCES = [
    "I really enjoyed that concert last night.",
    "What do you think about the new policy?",
    "This is the best pizza I have ever eaten.",
    "Have you seen the latest episode?",
    "I wonder what tomorrow will bring.",
]

ALL_SENTENCES = CLAIM_SENTENCES + NON_CLAIM_SENTENCES


class ClaimAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        sentence = random.choice(ALL_SENTENCES)
        self.client.post("/predict", json={"sentence": sentence}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
