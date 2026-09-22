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

# ── Modèles HuggingFace ────────────────────────────────────────────────────
LANG_MODEL = "papluca/xlm-roberta-base-language-detection"

TRANSLATE_MODELS = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
}

# ── Modèles financiers spécialisés ────────────────────────────────────────
#
# F1 : FOMC-RoBERTa — Politique monétaire des banques centrales
#   Entraîné sur les minutes FOMC (Fed) et communiqués BCE
#   Classifie : hawkish (restrictif) / dovish (accommodant) / neutral
#   Utilisation : analyse du contexte macro → signal pour les conseils
#
MONETARY_POLICY_MODEL = "gtfintechlab/FOMC-RoBERTa"
#
# F2 : FinBERT-tone topic detection — Type d'événement financier
#   Classifie la NEWS par CATÉGORIE : earnings / merger / analyst / regulatory /
#   stock_price / product / macro / personnel (changement de direction)
#   Utilisation : savoir POURQUOI une news est bonne/mauvaise, pas juste si elle l'est
#
FINANCIAL_TOPIC_MODEL = "nickmuchi/finbert-tone-finetuned-finance-topic-detection"
#
# F3 : RoBERTa-large sentiment financier — Sentiment renforcé (ensemble)
#   Fine-tuné sur corpus Reuters + Bloomberg (plus large que FinBERT de base)
#   Utilisé en ENSEMBLE avec ProsusAI/finbert : vote croisé → confiance plus fiable
#
FINANCIAL_SENTIMENT_STRONG = "nickmuchi/financial-roberta-large-sentiment-analysis"
FINANCIAL_SENTIMENT_BASE   = "ProsusAI/finbert"

# ── Labels de sortie des modèles financiers ───────────────────────────────
MONETARY_LABELS = {
    "hawkish": {
        "fr": "restrictive (hausse des taux probable) 🔴",
        "en": "hawkish (rate hike likely) 🔴",
        "de": "restriktiv (Zinserhöhung wahrscheinlich) 🔴",
        "es": "restrictiva (subida de tipos probable) 🔴",
        "it": "restrittiva (aumento dei tassi probabile) 🔴",
    },
    "dovish": {
        "fr": "accommodante (baisse des taux probable) 🟢",
        "en": "dovish (rate cut likely) 🟢",
        "de": "akkommodierend (Zinssenkung wahrscheinlich) 🟢",
        "es": "acomodaticia (bajada de tipos probable) 🟢",
        "it": "accomodante (taglio dei tassi probabile) 🟢",
    },
    "neutral": {
        "fr": "neutre ⚪",
        "en": "neutral ⚪",
        "de": "neutral ⚪",
        "es": "neutral ⚪",
        "it": "neutrale ⚪",
    },
}

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


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLES FINANCIERS SPÉCIALISÉS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_monetary_policy(text: str, lang: str = "fr") -> dict:
    """
    F1 — FOMC-RoBERTa : stance de politique monétaire.

    Entraîné sur les minutes FOMC et communiqués des banques centrales.
    Classifie n'importe quel texte sur la politique monétaire :
    - hawkish : ton restrictif → préoccupations sur l'inflation, taux à la hausse
    - dovish  : ton accommodant → soutien à la croissance, taux à la baisse
    - neutral : aucune orientation marquée

    API type : text-classification (TYPE 3 HF)
    Payload  : {"inputs": "The Fed remains committed to fighting inflation..."}
    Retour   : [{"label": "hawkish", "score": 0.87}, ...]

    Usage dans Folio : ajoute un signal macro "stance banque centrale" qui
    influence les recommandations (hawkish → prudence sur actions de croissance,
    dovish → favorable aux actifs risqués).
    """
    if not text or len(text.strip()) < 20:
        return {"stance": "neutral", "score": 0.5, "label_display": MONETARY_LABELS["neutral"].get(lang, "neutral ⚪")}

    result = _hf_post(MONETARY_POLICY_MODEL, {"inputs": text[:512]}, timeout=20)

    stance = "neutral"
    score = 0.5

    if isinstance(result, list) and result:
        scores = result[0] if isinstance(result[0], list) else result
        try:
            # Normaliser les labels (le modèle peut retourner "HAWKISH", "hawkish", etc.)
            normalized = [
                {"label": s["label"].lower().strip(), "score": s["score"]}
                for s in scores if "label" in s
            ]
            best = max(normalized, key=lambda x: x["score"])
            if best["label"] in MONETARY_LABELS:
                stance = best["label"]
                score = best["score"]
        except Exception:
            pass

    return {
        "stance": stance,
        "score": round(score, 3),
        "label_display": MONETARY_LABELS.get(stance, MONETARY_LABELS["neutral"]).get(lang, stance),
    }


def classify_financial_topic(text_en: str) -> str:
    """
    F2 — FinBERT-tone topic detection : type d'événement financier.

    Identifie la CATÉGORIE de l'actualité financière :
    - Earnings         : résultats trimestriels, bénéfices
    - Mergers/Acquisitions : fusions, rachats, OPA
    - Analyst Update   : objectif de cours, notation analyste
    - Regulatory/Legal : amende, régulation, procès
    - Stock Price      : variation de cours, volume
    - Product/Service  : lancement produit, innovation
    - Macro/Economy    : PIB, emploi, taux directeurs
    - Personnel Change : changement de direction, PDG

    API type : text-classification (TYPE 3 HF)
    Payload  : {"inputs": "Apple beats Q3 earnings estimates by 12%"}
    Retour   : [{"label": "Earnings", "score": 0.92}, ...]

    Usage dans Folio : contextualize l'actualité → "Apple a dépassé ses prévisions
    de résultats (+12%)" est bien plus informatif que juste "actualité positive".
    """
    if not text_en:
        return "general"

    result = _hf_post(FINANCIAL_TOPIC_MODEL, {"inputs": text_en[:512]}, timeout=20)

    if isinstance(result, list) and result:
        scores = result[0] if isinstance(result[0], list) else result
        try:
            best = max(scores, key=lambda x: x["score"])
            return best.get("label", "general").lower().replace(" ", "_")
        except Exception:
            pass
    return "general"


def analyze_sentiment_ensemble(text_en: str) -> dict:
    """
    F3 — Sentiment financier ensemble (vote croisé F3 + FinBERT de base).

    Utilise DEUX modèles de sentiment financier en parallèle :
    - F3 : nickmuchi/financial-roberta-large (entraîné Reuters/Bloomberg)
    - Base : ProsusAI/finbert (référence standard)

    Les deux votent → score moyen pondéré → confiance plus fiable qu'un seul modèle.

    Exemple :
    - FinBERT dit positive (0.72) → score: +0.72
    - RoBERTa dit positive (0.89) → score: +0.89
    → Consensus : positive avec confiance 0.81 (forte)

    Si les deux divergent, confiance faible → signal d'ambiguïté utile.

    API type : text-classification (TYPE 3 HF) — même format pour les deux modèles
    """
    label_map = {
        "bullish": "positive", "bearish": "negative",
        "positive": "positive", "negative": "negative", "neutral": "neutral",
        "topic-news-finance": "neutral",
    }
    scores_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    votes = []

    # F3 : RoBERTa-large financier (vote 1 — poids plus élevé : mieux entraîné)
    result_strong = _hf_post(FINANCIAL_SENTIMENT_STRONG, {"inputs": text_en[:512]}, timeout=25)
    if isinstance(result_strong, list) and result_strong:
        s_list = result_strong[0] if isinstance(result_strong[0], list) else result_strong
        try:
            best = max(s_list, key=lambda x: x["score"])
            normalized_label = label_map.get(best["label"].lower(), "neutral")
            votes.append({"label": normalized_label, "score": best["score"], "weight": 1.5})
        except Exception:
            pass

    # Base : ProsusAI/finbert (vote 2 — poids standard)
    result_base = _hf_post(FINANCIAL_SENTIMENT_BASE, {"inputs": text_en[:512]}, timeout=25)
    if isinstance(result_base, list) and result_base:
        s_list = result_base[0] if isinstance(result_base[0], list) else result_base
        try:
            best = max(s_list, key=lambda x: x["score"])
            normalized_label = label_map.get(best["label"].lower(), "neutral")
            votes.append({"label": normalized_label, "score": best["score"], "weight": 1.0})
        except Exception:
            pass

    if not votes:
        return {"label": "neutral", "score": 0.5, "confidence": 0.0, "consensus": False}

    # Score pondéré
    total_weight = sum(v["weight"] for v in votes)
    weighted_sum = sum(scores_map[v["label"]] * v["score"] * v["weight"] for v in votes)
    avg = weighted_sum / total_weight

    if avg > 0.15:
        label = "positive"
    elif avg < -0.15:
        label = "negative"
    else:
        label = "neutral"

    # Consensus : les deux modèles sont d'accord ?
    consensus = len(set(v["label"] for v in votes)) == 1 if len(votes) > 1 else True

    return {
        "label": label,
        "score": round((avg + 1) / 2, 3),   # normalisé 0–1
        "confidence": round(abs(avg), 3),
        "consensus": consensus,               # True si les 2 modèles s'accordent
    }


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
    Analyse de sentiment financier — utilise l'ensemble F3+FinBERT quand disponible.
    Retourne {"label": "positive|negative|neutral", "score": float}
    """
    result = analyze_sentiment_ensemble(text_en)
    return {"label": result["label"], "score": result["score"]}


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
    topics_seen: dict[str, int] = {}   # comptage des topics

    for art in articles[:8]:
        title = art.get("title", "") or ""
        desc = art.get("description", "") or ""
        combined = f"{title}. {desc}"[:400]
        if not combined.strip():
            continue
        headlines.append(title[:120])

        # F3 + FinBERT ensemble : sentiment avec vote croisé
        s_ensemble = analyze_sentiment_ensemble(combined)
        sentiments.append(s_ensemble)

        # F2 : classification du type d'événement financier
        topic = classify_financial_topic(combined)
        topics_seen[topic] = topics_seen.get(topic, 0) + 1

    if not sentiments:
        return {
            "ticker": ticker,
            "sentiment": "neutral",
            "score": 0.5,
            "confidence": 0.0,
            "consensus_rate": 0.0,
            "articles_analyzed": 0,
            "main_topic": "general",
            "headlines": [],
            "summary": _no_news_summary(ticker, user_lang),
        }

    # Agréger sentiment — pondéré par confiance de chaque article
    scores_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    total_conf = sum(s["confidence"] + 0.01 for s in sentiments)
    weighted_sum = sum(
        scores_map[s["label"]] * s["score"] * (s["confidence"] + 0.01)
        for s in sentiments
    )
    count = len(sentiments)
    avg = weighted_sum / total_conf

    if avg > 0.15:
        label = "positive"
    elif avg < -0.15:
        label = "negative"
    else:
        label = "neutral"

    confidence = abs(avg)
    score_normalized = (avg + 1) / 2
    consensus_rate = sum(1 for s in sentiments if s.get("consensus", True)) / count

    # Topic dominant parmi les articles analysés
    main_topic = max(topics_seen, key=topics_seen.get) if topics_seen else "general"

    summary = _build_summary(
        ticker, company_name, label, confidence, count, user_lang,
        main_topic=main_topic
    )

    return {
        "ticker": ticker,
        "sentiment": label,
        "score": round(score_normalized, 3),
        "confidence": round(confidence, 3),
        "consensus_rate": round(consensus_rate, 2),  # % articles où F3 et FinBERT s'accordent
        "articles_analyzed": count,
        "main_topic": main_topic,     # type d'événement dominant (earnings, M&A, regulatory...)
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
    main_topic: str = "general",
) -> str:
    label_str = SENTIMENT_LABELS.get(label, {}).get(lang, label)
    conf_pct = round(confidence * 100)

    # Traduction du topic dominant
    topic_labels = {
        "earnings": {"fr": "résultats", "en": "earnings", "de": "Ergebnisse", "es": "resultados", "it": "utili"},
        "mergers_acquisitions": {"fr": "fusion/acquisition", "en": "M&A", "de": "Fusion/Übernahme", "es": "fusión/adquisición", "it": "fusione/acquisizione"},
        "analyst_update": {"fr": "recommandation analyste", "en": "analyst update", "de": "Analysten-Update", "es": "actualización analista", "it": "aggiornamento analista"},
        "regulatory_legal": {"fr": "réglementation", "en": "regulatory", "de": "Regulierung", "es": "regulatorio", "it": "normativa"},
        "stock_price": {"fr": "cours boursier", "en": "stock price", "de": "Aktienkurs", "es": "cotización", "it": "prezzo azionario"},
        "product_service": {"fr": "produit/service", "en": "product/service", "de": "Produkt/Dienstleistung", "es": "producto/servicio", "it": "prodotto/servizio"},
        "macro": {"fr": "macro-économie", "en": "macroeconomics", "de": "Makroökonomie", "es": "macroeconomía", "it": "macroeconomia"},
        "personnel": {"fr": "changement de direction", "en": "leadership change", "de": "Führungswechsel", "es": "cambio directivo", "it": "cambio di leadership"},
    }
    topic_str = topic_labels.get(main_topic, {}).get(lang, main_topic.replace("_", " ")) if main_topic != "general" else ""
    topic_part = {"fr": f" Sujet dominant : {topic_str}.", "en": f" Main topic: {topic_str}.", "de": f" Hauptthema: {topic_str}.", "es": f" Tema principal: {topic_str}.", "it": f" Argomento principale: {topic_str}."}.get(lang, "") if topic_str else ""

    summaries = {
        "fr": (
            f"Sur {count} articles analysés (2 modèles FinBERT en consensus), le sentiment pour {name} ({ticker}) "
            f"est {label_str} (confiance : {conf_pct}%).{topic_part}"
        ),
        "en": (
            f"Across {count} articles analyzed (2-model FinBERT consensus), sentiment for {name} ({ticker}) "
            f"is {label_str} (confidence: {conf_pct}%).{topic_part}"
        ),
        "de": (
            f"Über {count} analysierte Artikel (2-Modell-Konsens): Das Sentiment für {name} ({ticker}) "
            f"ist {label_str} (Konfidenz: {conf_pct}%).{topic_part}"
        ),
        "es": (
            f"Según {count} artículos analizados (consenso 2 modelos), el sentimiento de {name} ({ticker}) "
            f"es {label_str} (confianza: {conf_pct}%).{topic_part}"
        ),
        "it": (
            f"Su {count} articoli analizzati (consenso 2 modelli), il sentiment per {name} ({ticker}) "
            f"è {label_str} (confidenza: {conf_pct}%).{topic_part}"
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
