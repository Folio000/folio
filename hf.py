import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

def analyze_sentiment(text: str) -> dict:
    response = requests.post(
     "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": text},
        timeout=30
    )
    scores = response.json()[0]
    best = max(scores, key=lambda x: x["score"])
    return best