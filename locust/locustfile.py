"""Flood test: users mostly send leaf photos to /predict and sometimes check /status.

locust -f locust/locustfile.py --host http://localhost:8080 --headless -u 30 -r 10 -t 60s
"""

import glob
import os
import random

from locust import HttpUser, between, task

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = sorted(glob.glob(os.path.join(HERE, "..", "sample_images", "predict", "*", "*.jpg")))
if not SAMPLES:
    raise SystemExit("No images found in sample_images/predict/<class>/")
IMAGES = [(os.path.basename(p), open(p, "rb").read()) for p in SAMPLES]


class LeafUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(8)
    def predict(self):
        name, data = random.choice(IMAGES)
        with self.client.post(
            "/predict", files={"file": (name, data, "image/jpeg")}, name="/predict", catch_response=True
        ) as r:
            if r.status_code != 200 or "predicted_class" not in r.text:
                r.failure(f"bad response {r.status_code}")

    @task(2)
    def status(self):
        self.client.get("/status", name="/status")

    @task(1)
    def class_distribution(self):
        self.client.get("/visualizations/class-distribution", name="/visualizations/class-distribution")
