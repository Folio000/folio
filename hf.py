import requests
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

TRANSLATE_MODELS = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
}

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
        return lang if lang in TRANSLATE_MODELS or lang == "en" else "en"
    except Exception:
        return "en"

def translate_to_english(text: str, lang: str) -> str:
    model = TRANSLATE_MODELS.get(lang)
    if not model:
        return text
    try:
        r = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{model}",
            headers=HEADERS,
            json={"inputs": text},
            timeout=20
        )
        result = r.json()
        if isinstance(result, list) and result:
            return result[0].get("translation_text", text)
        return text
    except Exception:
        return text

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
