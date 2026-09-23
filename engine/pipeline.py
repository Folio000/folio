"""
engine/pipeline.py — Orchestration complète d'une question Folio v7

Flux :
1. Détection de la langue
2. Extraction du ticker mentionné
3. Enrichissement parallèle (marché + news + macro)
4. Construction du contexte
5. Appel LLM (Groq)
6. Retour structuré
"""

import re
import concurrent.futures
from langdetect import detect

from data.market import get_quote, get_snapshot
from data.news import get_news_context
from data.macro import get_macro_summary
from engine.llm import ask

# Tickers reconnus (étendable)
TICKER_ALIASES: dict[str, str] = {
    "apple": "AAPL", "nvidia": "NVDA", "nvda": "NVDA",
    "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "tesla": "TSLA",
    "netflix": "NFLX", "paypal": "PYPL", "spotify": "SPOT",
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "or": "GC=F", "gold": "GC=F", "pétrole": "CL=F", "oil": "CL=F",
    "sp500": "^GSPC", "s&p": "^GSPC", "nasdaq": "^IXIC",
    "cac": "^FCHI", "cac40": "^FCHI",
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "berkshire": "BRK-B",
    "lvmh": "MC.PA", "airbus": "AIR.PA", "bnp": "BNP.PA",
    "totalenergies": "TTE.PA", "total": "TTE.PA",
    "arm": "ARM", "amd": "AMD", "intel": "INTC",
    "tsmc": "TSM", "samsung": "005930.KS",
}

# Regex : ticker en majuscules (ex: AAPL, NVDA, BTC-USD)
_TICKER_RE = re.compile(r"\b([A-Z]{2,5}(?:-USD)?)\b")


def _detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "fr"


def _extract_ticker(text: str) -> str | None:
    lower = text.lower()
    for alias, ticker in TICKER_ALIASES.items():
        if alias in lower:
            return ticker
    # Cherche un ticker en majuscules dans le texte brut
    matches = _TICKER_RE.findall(text)
    if matches:
        return matches[0]
    return None


def _build_context(ticker: str | None, lang: str) -> str:
    """
    Construit le bloc de contexte injecté dans le prompt LLM.
    Les appels yfinance / RSS / FRED tournent en parallèle.
    """
    parts: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        fut_macro = pool.submit(get_macro_summary)

        if ticker:
            fut_quote = pool.submit(get_quote, ticker)
            fut_news = pool.submit(get_news_context, ticker)
        else:
            fut_quote = None
            fut_news = pool.submit(get_news_context, None)

        macro = fut_macro.result(timeout=10)
        if macro:
            parts.append(f"[Macro]\n{macro}")

        if fut_quote:
            quote = fut_quote.result(timeout=10)
            if quote:
                parts.append(f"[Marché — {ticker}]\n{quote}")

        news = fut_news.result(timeout=10)
        if news:
            parts.append(f"[Actualités récentes]\n{news}")

    if not parts:
        return "Aucune donnée de marché disponible pour le moment."

    return "\n\n".join(parts)


def run(question: str) -> dict:
    """
    Point d'entrée principal du pipeline.
    Retourne un dict avec : answer, ticker, lang.
    """
    lang = _detect_lang(question)
    ticker = _extract_ticker(question)

    context = _build_context(ticker, lang)
    answer = ask(question, context)

    return {
        "answer": answer,
        "ticker": ticker,
        "lang": lang,
        "data_available": bool(context),
    }
