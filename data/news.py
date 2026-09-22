"""
data/news.py — Pipeline NLP multilingue sur actualités financières
Sources : NewsAPI + Yahoo Finance news
Pipeline : texte → détection langue → traduction EN → FinBERT → score agrégé
Réponse : dans la langue de l'utilisateur (FR / DE / ES / IT / EN)
"""
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Modèles HuggingFace
LANG_MODEL = "papluca/xlm-roberta-base-language-detection"
TRANSLATE_MODELS = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
}
SENTIMENT_MODELS = [
    "nickmuchi/finbert-tone-finetuned-finance-topic-detection",
    "ProsusAI/finbert",
]

# Libellés pour la réponse dans chaque langue
SENTIMENT_LABELS = {
    "positive": {
        "fr": "positif 📈", "en": "positive 📈",
        "de": "positiv 📈", "es": "positivo 📈", "it": "positivo 📈",
    },
    "negative": {
        "fr": "négatif 📉", "en": "negative 📉",
        "de": "negativ 📉", "es": "negativo 📉", "it": "negativo 📉",
    },
    "neutral": {
        "fr": "neutre ➡️", "en": "neutral ➡️",
        "de": "neutral ➡️", "es": "neutral ➡️", "it": "neutro ➡️",
    },
}

SUMMARY_LABELS = {
    "fr": "Résumé actualités",
    "en": "News summary",
    "de": "Nachrichtenübersicht",
    "es": "Resumen de noticias",
    "it": "Riepilogo notizie",
}


def _hf_post(model: str, payload: dict, timeout: int = 20) -> dict | list | None:
    try:
        r = requests.post(
            f"https://router.huggingface.co/hf-inference/models/{model}",
            headers=HF_HEADERS,
            json=payload,
            timeout=timeout,
        )
        return r.json()
    except Exception:
        return None


def detect_language(text: str) -> str:
    """Détecte la langue du texte. Retourne 'en' par défaut."""
    result = _hf_post(LANG_MODEL, {"inputs": text[:200]})
    if isinstance(result, list) and result and isinstance(result[0], list):
        scores = result[0]
    elif isinstance(result, list) and result and isinstance(result[0], dict):
        scores = result
    else:
        return "en"
    try:
        best = max(scores, key=lambda x: x["score"])
        lang = best["label"].lower()
        return lang if lang in TRANSLATE_MODELS or lang == "en" else "en"
    except Exception:
        return "en"


def translate_to_english(text: str, lang: str) -> str:
    """Traduit vers l'anglais si nécessaire."""
    if lang == "en" or lang not in TRANSLATE_MODELS:
        return text
    model = TRANSLATE_MODELS[lang]
    result = _hf_post(model, {"inputs": text}, timeout=25)
    if isinstance(result, list) and result:
        return result[0].get("translation_text", text)
    return text


def analyze_sentiment(text_en: str) -> dict:
    """
    Analyse de sentiment FinBERT sur texte en anglais.
    Retourne {"label": "positive|negative|neutral", "score": float}
    """
    label_map = {
        "bullish": "positive", "bearish": "negative",
        "positive": "positive", "negative": "negative", "neutral": "neutral",
        "topic-news-finance": "neutral",
    }
    for model in SENTIMENT_MODELS:
        result = _hf_post(model, {"inputs": text_en[:512]}, timeout=30)
        if not result:
            continue
        scores = None
        if isinstance(result, list) and result:
            scores = result[0] if isinstance(result[0], list) else result
        if not scores:
            continue
        try:
            normalized = [
                {"label": label_map.get(s["label"].lower(), "neutral"), "score": s["score"]}
                for s in scores if "label" in s
            ]
            best = max(normalized, key=lambda x: x["score"])
            return best
        except Exception:
            continue
    return {"label": "neutral", "score": 0.5}


def _fetch_newsapi(query: str, lang_code: str = "en", days: int = 3) -> list[dict]:
    """Récupère des articles NewsAPI pour une requête donnée."""
    if not NEWS_API_KEY:
        return []
    # NewsAPI supporte : en, fr, de, es, it
    supported = {"en", "fr", "de", "es", "it"}
    api_lang = lang_code if lang_code in supported else "en"
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": api_lang,
                "from": from_date,
                "sortBy": "relevancy",
                "pageSize": 10,
                "apiKey": NEWS_API_KEY,
            },
            timeout=10,
        )
        data = r.json()
        return data.get("articles", [])
    except Exception:
        return []


def _fetch_newsapi_headlines(category: str = "business", lang_code: str = "en") -> list[dict]:
    """Récupère les titres de la presse financière (top headlines)."""
    if not NEWS_API_KEY:
        return []
    supported_countries = {"fr", "de", "es", "it"}
    country = lang_code if lang_code in supported_countries else "us"
    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={
                "category": category,
                "country": country,
                "pageSize": 8,
                "apiKey": NEWS_API_KEY,
            },
            timeout=10,
        )
        data = r.json()
        return data.get("articles", [])
    except Exception:
        return []


def get_ticker_news_sentiment(
    ticker: str,
    company_name: str,
    user_lang: str = "en",
) -> dict:
    """
    Score de sentiment agrégé pour un ticker basé sur les actualités récentes.
    Retourne :
    {
        "ticker": str,
        "sentiment": "positive|negative|neutral",
        "score": float,
        "confidence": float,
        "articles_analyzed": int,
        "headlines": list[str],  # dans la langue de NewsAPI
        "summary": str,          # explication en user_lang
    }
    """
    # Cherche en anglais pour toucher plus d'articles
    articles = _fetch_newsapi(f"{company_name} stock", lang_code="en", days=5)
    if not articles:
        articles = _fetch_newsapi(ticker, lang_code="en", days=7)

    sentiments = []
    headlines = []

    for art in articles[:8]:
        title = art.get("title", "") or ""
        desc = art.get("description", "") or ""
        combined = f"{title}. {desc}"[:400]
        if not combined.strip():
            continue
        headlines.append(title[:120])

        # Le texte est en anglais (NewsAPI EN) — on analyse directement
        s = analyze_sentiment(combined)
        sentiments.append(s)

    if not sentiments:
        return {
            "ticker": ticker,
            "sentiment": "neutral",
            "score": 0.5,
            "confidence": 0.0,
            "articles_analyzed": 0,
            "headlines": [],
            "summary": _no_news_summary(ticker, user_lang),
        }

    # Agréger : moyenne pondérée
    scores_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    weighted_sum = sum(scores_map[s["label"]] * s["score"] for s in sentiments)
    count = len(sentiments)
    avg = weighted_sum / count

    if avg > 0.15:
        label = "positive"
    elif avg < -0.15:
        label = "negative"
    else:
        label = "neutral"

    confidence = abs(avg)
    score_normalized = (avg + 1) / 2  # 0–1

    summary = _build_summary(ticker, company_name, label, confidence, count, user_lang)

    return {
        "ticker": ticker,
        "sentiment": label,
        "score": round(score_normalized, 3),
        "confidence": round(confidence, 3),
        "articles_analyzed": count,
        "headlines": headlines[:5],
        "summary": summary,
    }


def get_market_news(user_lang: str = "fr", max_headlines: int = 6) -> list[dict]:
    """
    Actualités financières générales dans la langue de l'utilisateur.
    Retourne liste de {title, source, url, sentiment_label, sentiment_score}
    """
    articles = _fetch_newsapi_headlines(category="business", lang_code=user_lang)
    if not articles:
        articles = _fetch_newsapi("finance bourse investissement", lang_code=user_lang, days=2)

    result = []
    for art in articles[:max_headlines]:
        title = art.get("title", "") or ""
        if not title:
            continue
        desc = art.get("description", "") or ""
        combined = f"{title}. {desc}"[:400]

        # Si article dans langue de l'utilisateur, traduire d'abord pour FinBERT
        if user_lang != "en":
            combined_en = translate_to_english(combined, user_lang)
        else:
            combined_en = combined

        s = analyze_sentiment(combined_en)

        result.append({
            "title": title[:150],
            "source": art.get("source", {}).get("name", ""),
            "url": art.get("url", ""),
            "published_at": art.get("publishedAt", "")[:10],
            "sentiment": s["label"],
            "sentiment_score": round(s["score"], 3),
        })

    return result


def _build_summary(
    ticker: str,
    name: str,
    label: str,
    confidence: float,
    count: int,
    lang: str,
) -> str:
    label_str = SENTIMENT_LABELS.get(label, {}).get(lang, label)
    conf_pct = round(confidence * 100)

    summaries = {
        "fr": (
            f"Sur {count} articles analysés, le sentiment pour {name} ({ticker}) "
            f"est {label_str} (confiance : {conf_pct}%)."
        ),
        "en": (
            f"Across {count} articles analyzed, sentiment for {name} ({ticker}) "
            f"is {label_str} (confidence: {conf_pct}%)."
        ),
        "de": (
            f"Über {count} analysierte Artikel: Das Sentiment für {name} ({ticker}) "
            f"ist {label_str} (Konfidenz: {conf_pct}%)."
        ),
        "es": (
            f"Según {count} artículos analizados, el sentimiento de {name} ({ticker}) "
            f"es {label_str} (confianza: {conf_pct}%)."
        ),
        "it": (
            f"Su {count} articoli analizzati, il sentiment per {name} ({ticker}) "
            f"è {label_str} (confidenza: {conf_pct}%)."
        ),
    }
    return summaries.get(lang, summaries["en"])


def _no_news_summary(ticker: str, lang: str) -> str:
    msgs = {
        "fr": f"Aucune actualité récente trouvée pour {ticker}.",
        "en": f"No recent news found for {ticker}.",
        "de": f"Keine aktuellen Nachrichten für {ticker} gefunden.",
        "es": f"No se encontraron noticias recientes para {ticker}.",
        "it": f"Nessuna notizia recente trovata per {ticker}.",
    }
    return msgs.get(lang, msgs["en"])
