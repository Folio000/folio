import requests
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def detect_language(text: str) -> str:
    try:
        r = requests.post(
            "https://router.huggingface.co/hf-inference/models/papluca/xlm-roberta-base-language-detection",
            headers=HEADERS,
            json={"inputs": text[:200]},
            timeout=15
        )
        scores = r.json()[0]
        best = max(scores, key=lambda x: x["score"])
        lang = best["label"].lower()
        return lang if lang in ("fr", "en") else "en"
    except Exception:
        return "en"

def analyze_sentiment(text: str) -> dict:
    r = requests.post(
        "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert",
        headers=HEADERS,
        json={"inputs": text},
        timeout=30
    )
    scores = r.json()[0]
    best = max(scores, key=lambda x: x["score"])
    return best
