"""
hf.py — Orchestrateur central Folio v6
Pipeline complet : données de marché + macro + news + technicals + portfolio → LLM
Multilingue : FR / DE / ES / IT / EN détecté automatiquement, réponse dans la langue de l'utilisateur
"""
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from data.fetcher import (
    get_universe_snapshot, get_ticker_full,
    get_news_headlines, extract_ticker,
    UNIVERSE, TICKER_MAP,
)
from data.news import (
    detect_language, translate_to_english,
    analyze_sentiment, get_ticker_news_sentiment,
    get_market_news,
)
from data.macro import get_macro_summary
from analysis.technicals import get_technicals, format_signal
from analysis.portfolio import analyze_portfolio, portfolio_summary_for_prompt

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Réexporter pour compatibilité avec main.py
__all__ = [
    "detect_language", "translate_to_english", "analyze_sentiment",
    "extract_ticker", "get_market_data", "get_universe_snapshot",
    "advise", "get_ticker_full", "get_portfolio_analysis",
    "get_market_news", "UNIVERSE", "TICKER_MAP",
]


def get_market_data(ticker: str) -> dict | None:
    """Alias de fetcher.get_ticker_full pour compatibilité."""
    from data.fetcher import get_market_data as _gmd
    return _gmd(ticker)


def _llm_call(prompt: str, max_tokens: int = 500) -> str | None:
    """
    Appel LLM via HuggingFace Inference API.
    Essaie Mistral-7B puis Zephyr en fallback.
    """
    for model in [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "HuggingFaceH4/zephyr-7b-beta",
    ]:
        try:
            r = requests.post(
                f"https://router.huggingface.co/hf-inference/models/{model}",
                headers=HF_HEADERS,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": 0.4,
                        "return_full_text": False,
                        "do_sample": True,
                    }
                },
                timeout=65,
            )
            result = r.json()
            if isinstance(result, list) and result:
                text = result[0].get("generated_text", "").strip()
            elif isinstance(result, dict):
                text = result.get("generated_text", "").strip()
            else:
                continue
            if text and len(text) > 40:
                return text
        except Exception:
            continue
    return None


def _lang_instruction(lang: str) -> str:
    """Instruction de langue pour le prompt LLM."""
    instructions = {
        "fr": "Réponds en français, de manière directe et structurée.",
        "en": "Reply in English, direct and structured.",
        "de": "Antworte auf Deutsch, direkt und strukturiert.",
        "es": "Responde en español, de forma directa y estructurada.",
        "it": "Rispondi in italiano, in modo diretto e strutturato.",
    }
    return instructions.get(lang, "Reply in the same language as the question.")


def _fallback_response(snapshot: list[dict], lang: str, amount: str | None) -> str:
    """Réponse template si le LLM échoue."""
    top3 = snapshot[:3]
    amount_str = f"{amount}€ " if amount else ""

    if lang == "fr":
        lines = [f"Recommandation basée sur les données de marché actuelles ({amount_str}):"]
        for d in top3:
            sign = "+" if d["change_pct"] >= 0 else ""
            lines.append(f"• {d['name']} ({d['ticker']}) — {d['price']} {d.get('currency','EUR')} ({sign}{d['change_pct']}%)")
    elif lang == "de":
        lines = [f"Empfehlung basierend auf aktuellen Marktdaten ({amount_str}):"]
        for d in top3:
            sign = "+" if d["change_pct"] >= 0 else ""
            lines.append(f"• {d['name']} ({d['ticker']}) — {d['price']} {d.get('currency','')} ({sign}{d['change_pct']}%)")
    elif lang == "es":
        lines = [f"Recomendación basada en datos de mercado actuales ({amount_str}):"]
        for d in top3:
            sign = "+" if d["change_pct"] >= 0 else ""
            lines.append(f"• {d['name']} ({d['ticker']}) — {d['price']} {d.get('currency','')} ({sign}{d['change_pct']}%)")
    elif lang == "it":
        lines = [f"Raccomandazione basata sui dati di mercato attuali ({amount_str}):"]
        for d in top3:
            sign = "+" if d["change_pct"] >= 0 else ""
            lines.append(f"• {d['name']} ({d['ticker']}) — {d['price']} {d.get('currency','')} ({sign}{d['change_pct']}%)")
    else:
        lines = [f"Recommendation based on current market data ({amount_str}):"]
        for d in top3:
            sign = "+" if d["change_pct"] >= 0 else ""
            lines.append(f"• {d['name']} ({d['ticker']}) — {d['price']} {d.get('currency','')} ({sign}{d['change_pct']}%)")

    return "\n".join(lines)


def advise(
    question: str,
    portfolio: list | None = None,
    risk_profile: str = "modéré",
) -> dict:
    """
    Conseiller Folio v6 — pipeline complet multilingue.

    1. Détection de langue
    2. Snapshot marché + macro + news (en parallèle)
    3. Analyse technique du ticker mentionné
    4. Analyse de portefeuille si fourni
    5. Construction du prompt avec tout le contexte
    6. Appel LLM (Mistral-7B → Zephyr fallback)
    7. Réponse dans la langue de l'utilisateur

    Retourne :
    {
        answer, lang, amount, snapshot, mentioned_assets,
        news_sentiment, macro_context, technical, portfolio_analysis
    }
    """
    portfolio = portfolio or []

    # 1. Détection de langue
    lang = detect_language(question)

    # 2. Extraction montant
    amount_match = re.search(
        r'(\d[\d\s]*)\s*[€$£]|[€$£]\s*(\d[\d\s]*)|(\d[\d\s]+)\s*(?:euros?|dollar|pound|livre)',
        question, re.IGNORECASE
    )
    amount = None
    if amount_match:
        raw = amount_match.group(1) or amount_match.group(2) or amount_match.group(3)
        amount = raw.replace(" ", "").strip() if raw else None

    # 3. Appels parallèles : snapshot marché + macro + news marché
    snapshot = []
    macro_ctx = ""
    market_news = []
    technical = {}

    def _fetch_snapshot():
        return get_universe_snapshot(max_tickers=16)

    def _fetch_macro():
        return get_macro_summary(lang=lang)

    def _fetch_news():
        return get_market_news(user_lang=lang, max_headlines=5)

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_snap = ex.submit(_fetch_snapshot)
        f_macro = ex.submit(_fetch_macro)
        f_news = ex.submit(_fetch_news)

        snapshot = f_snap.result()
        macro_ctx = f_macro.result()
        market_news = f_news.result()

    # 4. Ticker mentionné → analyse technique + sentiment news
    ticker_mentioned = extract_ticker(question)
    news_sentiment = None

    if ticker_mentioned:
        name = UNIVERSE.get(ticker_mentioned, ticker_mentioned)
        technical = get_technicals(ticker_mentioned)
        news_sentiment = get_ticker_news_sentiment(
            ticker_mentioned, name, user_lang=lang
        )

    # 5. Analyse portefeuille
    portfolio_analysis = None
    portfolio_block = ""
    if portfolio:
        try:
            portfolio_analysis = analyze_portfolio(portfolio)
            portfolio_block = portfolio_summary_for_prompt(portfolio, lang=lang)
        except Exception:
            portfolio_block = _simple_portfolio_text(portfolio, lang)
    else:
        portfolio_block = _no_portfolio_text(lang)

    # 6. Construction du prompt
    market_lines = []
    for d in snapshot[:10]:
        sign = "+" if d["change_pct"] >= 0 else ""
        rec = f" [{d.get('recommendation','').upper()}]" if d.get("recommendation") else ""
        market_lines.append(
            f"  {d['name']} ({d['ticker']}): {d['price']} {d.get('currency','EUR')}"
            f" ({sign}{d['change_pct']}%){rec}"
        )

    news_block = ""
    if market_news:
        sentiments_emoji = {"positive": "📈", "negative": "📉", "neutral": "➡️"}
        news_lines = [
            f"  {sentiments_emoji.get(n['sentiment'],'➡️')} {n['title'][:100]}"
            for n in market_news[:4]
        ]
        news_block = "Actualités marché :\n" + "\n".join(news_lines)

    tech_block = ""
    if technical and "error" not in technical:
        tech_block = f"Analyse technique {ticker_mentioned}: " + format_signal(technical, lang=lang)
        if news_sentiment:
            tech_block += f"\nSentiment actualités {ticker_mentioned}: {news_sentiment['summary']}"

    amount_line = f"Montant à investir : {amount}€\n" if amount else ""

    prompt = (
        f"<s>[INST] Tu es Folio, un conseiller financier expert, direct et factuel. "
        f"Profil investisseur : {risk_profile}.\n\n"
        f"Question : \"{question}\"\n\n"
        f"Données de marché temps réel :\n" + "\n".join(market_lines) + "\n\n"
        f"Contexte macro : {macro_ctx}\n\n"
        + (f"{tech_block}\n\n" if tech_block else "")
        + (f"{news_block}\n\n" if news_block else "")
        + f"{amount_line}"
        + f"{portfolio_block}\n\n"
        f"Donne une recommandation directe et concrète (1-3 actifs spécifiques, "
        f"montants suggérés si applicable, justification courte basée sur les données). "
        f"Sois précis, factuel, professionnel. "
        f"{_lang_instruction(lang)} [/INST]"
    )

    # 7. Appel LLM
    recommendation = _llm_call(prompt, max_tokens=500)

    if not recommendation:
        recommendation = _fallback_response(snapshot, lang, amount)

    # Assets mentionnés dans la réponse
    mentioned = []
    for d in snapshot:
        if d["ticker"] in recommendation or d["name"].lower() in recommendation.lower():
            mentioned.append(d)

    return {
        "answer": recommendation,
        "lang": lang,
        "amount": amount,
        "snapshot": snapshot[:8],
        "mentioned_assets": mentioned[:4],
        "news_sentiment": news_sentiment,
        "macro_context": macro_ctx,
        "technical": technical if "error" not in technical else None,
        "portfolio_analysis": portfolio_analysis,
    }


def get_portfolio_analysis(holdings: list[dict]) -> dict:
    """Endpoint dédié à l'analyse de portefeuille."""
    if not holdings:
        return {"error": "Portefeuille vide"}
    return analyze_portfolio(holdings)


def _simple_portfolio_text(portfolio: list[dict], lang: str) -> str:
    lines = [f"  - {p.get('name', p.get('ticker', '?'))}: {p.get('amount', 0)}€" for p in portfolio]
    labels = {
        "fr": "Portefeuille actuel :",
        "en": "Current portfolio:",
        "de": "Aktuelles Portfolio:",
        "es": "Cartera actual:",
        "it": "Portafoglio attuale:",
    }
    return labels.get(lang, labels["en"]) + "\n" + "\n".join(lines)


def _no_portfolio_text(lang: str) -> str:
    msgs = {
        "fr": "Pas de portefeuille renseigné.",
        "en": "No portfolio provided.",
        "de": "Kein Portfolio angegeben.",
        "es": "Sin cartera definida.",
        "it": "Nessun portafoglio fornito.",
    }
    return msgs.get(lang, msgs["en"])
