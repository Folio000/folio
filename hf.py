import requests
import os
import re
import yfinance as yf
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

# Dictionnaire de noms -> tickers (FR/EN/DE/ES/IT)
TICKER_MAP = {
    # US
    "apple": "AAPL", "nvidia": "NVDA", "microsoft": "MSFT",
    "amazon": "AMZN", "tesla": "TSLA", "meta": "META",
    "alphabet": "GOOGL", "google": "GOOGL", "netflix": "NFLX",
    "jpmorgan": "JPM", "goldman sachs": "GS", "goldman": "GS",
    "jp morgan": "JPM", "bank of america": "BAC",
    # FR
    "lvmh": "MC.PA", "air liquide": "AI.PA", "totalenergies": "TTE.PA",
    "total": "TTE.PA", "bnp": "BNP.PA", "bnp paribas": "BNP.PA",
    "sanofi": "SAN.PA", "airbus": "AIR.PA", "kering": "KER.PA",
    "hermes": "RMS.PA", "hermès": "RMS.PA", "saint-gobain": "SGO.PA",
    "michelin": "ML.PA", "danone": "BN.PA", "renault": "RNO.PA",
    "stellantis": "STLAM.MI", "carrefour": "CA.PA",
    "société générale": "GLE.PA", "credit agricole": "ACA.PA",
    # DE
    "volkswagen": "VOW.DE", "vw": "VOW.DE", "deutsche bank": "DBK.DE",
    "siemens": "SIE.DE", "bmw": "BMW.DE", "mercedes": "MBG.DE",
    "bayer": "BAYN.DE", "basf": "BASF.DE", "sap": "SAP.DE",
    "allianz": "ALV.DE", "adidas": "ADS.DE",
    # IT
    "ferrari": "RACE.MI", "eni": "ENI.MI", "intesa": "ISP.MI",
    "unicredit": "UCG.MI", "enel": "ENEL.MI",
    # ES
    "inditex": "ITX.MC", "zara": "ITX.MC", "santander": "SAN.MC",
    "iberdrola": "IBE.MC", "bbva": "BBVA.MC",
    # UK
    "hsbc": "HSBA.L", "shell": "SHEL.L", "bp": "BP.L",
    "barclays": "BARC.L", "astrazeneca": "AZN.L",
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
    for model in [
        "nickmuchi/finbert-tone-finetuned-finance-topic-detection",
        "ProsusAI/finbert"
    ]:
        try:
            r = requests.post(
                f"https://router.huggingface.co/hf-inference/models/{model}",
                headers=HEADERS,
                json={"inputs": text},
                timeout=30
            )
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                scores = data[0]
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                scores = data
            else:
                continue
            # Normalise les labels (Bullish/Bearish -> positive/negative)
            label_map = {
                "bullish": "positive", "bearish": "negative",
                "positive": "positive", "negative": "negative", "neutral": "neutral"
            }
            normalized = []
            for s in scores:
                lbl = label_map.get(s["label"].lower(), "neutral")
                normalized.append({"label": lbl, "score": s["score"]})
            best = max(normalized, key=lambda x: x["score"])
            return best
        except Exception:
            continue
    return {"label": "neutral", "score": 0.5}


def extract_ticker(text: str) -> str | None:
    # 1. Symbole explicite $AAPL ou $MC.PA
    match = re.search(r'\$([A-Z]{1,5}(?:\.[A-Z]{1,2})?)', text)
    if match:
        return match.group(1)
    # 2. Nom de société dans le dictionnaire
    text_lower = text.lower()
    for name, ticker in sorted(TICKER_MAP.items(), key=lambda x: -len(x[0])):
        if name in text_lower:
            return ticker
    return None


def get_market_data(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        hist = t.history(period="2d")
        if hist.empty:
            return None
        last_price = float(hist["Close"].iloc[-1])
        prev_price = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_price
        change_pct = ((last_price - prev_price) / prev_price) * 100 if prev_price else 0

        # Données fondamentales (peut être vide pour certains tickers)
        full = {}
        try:
            full = t.info
        except Exception:
            pass

        return {
            "ticker": ticker,
            "name": full.get("shortName") or full.get("longName") or ticker,
            "price": round(last_price, 2),
            "change_pct": round(change_pct, 2),
            "currency": full.get("currency", info.currency if hasattr(info, "currency") else ""),
            "market_cap": full.get("marketCap"),
            "pe_ratio": full.get("trailingPE"),
            "week52_high": full.get("fiftyTwoWeekHigh"),
            "week52_low": full.get("fiftyTwoWeekLow"),
            "volume": full.get("volume"),
            "sector": full.get("sector"),
        }
    except Exception:
        return None
