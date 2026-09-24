"""
engine/pipeline.py — Orchestration Folio v8
Flux :
1. Détection langue + extraction ticker
2. Récupération mémoire (analyses passées du même ticker)
3. Enrichissement parallèle : marché + macro + RSS + scraper multi-sources
4. Construction contexte enrichi (mémoire + données fraîches)
5. Appel LLM Groq
6. Persistance de l'analyse en PostgreSQL (thread daemon — non bloquant)
"""

import re
import logging
import threading
import concurrent.futures
from langdetect import detect

from data.market import get_quote, get_snapshot
from data.news import get_news_context
from data.macro import get_macro_summary
from data.scraper import scrape_all, format_for_context
from engine.llm import ask
from engine import memory

logger = logging.getLogger(__name__)

# ── Aliases ticker ─────────────────────────────────────────────────────

TICKER_ALIASES: dict[str, str] = {
    # Tech US
    "apple": "AAPL", "nvidia": "NVDA", "nvda": "NVDA",
    "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
    "amazon": "AMZN", "meta": "META", "tesla": "TSLA",
    "netflix": "NFLX", "paypal": "PYPL", "spotify": "SPOT",
    "salesforce": "CRM", "adobe": "ADBE", "oracle": "ORCL",
    "uber": "UBER", "airbnb": "ABNB", "palantir": "PLTR",
    "arm": "ARM", "amd": "AMD", "intel": "INTC",
    "qualcomm": "QCOM", "tsmc": "TSM", "broadcom": "AVGO",
    # Finance US
    "jpmorgan": "JPM", "jp morgan": "JPM", "goldman": "GS",
    "berkshire": "BRK-B", "visa": "V", "mastercard": "MA",
    "blackrock": "BLK", "blackstone": "BX",
    # Crypto
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "xrp": "XRP-USD",
    # Matières premières
    "or": "GC=F", "gold": "GC=F", "pétrole": "CL=F", "oil": "CL=F",
    "argent": "SI=F", "silver": "SI=F", "cuivre": "HG=F",
    "gaz naturel": "NG=F",
    # Indices
    "sp500": "^GSPC", "s&p": "^GSPC", "s&p500": "^GSPC",
    "nasdaq": "^IXIC", "dow jones": "^DJI", "dow": "^DJI",
    "cac": "^FCHI", "cac40": "^FCHI", "cac 40": "^FCHI",
    "dax": "^GDAXI", "eurostoxx": "^STOXX50E",
    "nikkei": "^N225", "hang seng": "^HSI",
    # Obligations
    "treasury": "^TNX", "t-bond": "^TYX", "t-note": "^TNX",
    # Europe FR
    "lvmh": "MC.PA", "airbus": "AIR.PA", "bnp": "BNP.PA",
    "totalenergies": "TTE.PA", "total": "TTE.PA",
    "sanofi": "SAN.PA", "loreal": "OR.PA", "l'oréal": "OR.PA",
    "hermes": "RMS.PA", "hermès": "RMS.PA",
    "danone": "BN.PA", "renault": "RNO.PA", "stellantis": "STLAM.MI",
    "societe generale": "GLE.PA", "société générale": "GLE.PA",
    "axa": "CS.PA", "safran": "SAF.PA", "capgemini": "CAP.PA",
}

_TICKER_RE = re.compile(r"\b([A-Z]{2,5}(?:[.-][A-Z]{2,4})?)\b")


# ── Helpers ────────────────────────────────────────────────────────────

def _detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "fr"


def _extract_ticker(text: str) -> str | None:
    lower = text.lower()
    # Correspondance exacte sur les alias (plus long d'abord pour éviter "or" ⊂ "oracle")
    for alias in sorted(TICKER_ALIASES, key=len, reverse=True):
        if alias in lower:
            return TICKER_ALIASES[alias]
    # Ticker en majuscules dans le texte
    matches = _TICKER_RE.findall(text)
    if matches:
        return matches[0]
    return None


def _format_memory(past: list[dict]) -> str:
    """Formate les analyses passées pour le contexte LLM."""
    if not past:
        return ""
    lines = ["[Analyses précédentes — même valeur]"]
    for p in past:
        lines.append(f"• ({p['at']}) Q: {p['question'][:100]}")
        lines.append(f"  Résumé: {p['answer'][:300]}...")
    return "\n".join(lines)


def _fire_and_forget(fn, *args):
    """Lance fn(*args) dans un thread daemon — ne bloque jamais la réponse HTTP."""
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()


# ── Contexte ───────────────────────────────────────────────────────────

def _build_context(ticker: str | None, question: str, lang: str) -> str:
    """
    Assemble le contexte en parallèle :
    - mémoire PostgreSQL (analyses passées)
    - données marché (yfinance)
    - macro (FRED)
    - RSS existants
    - scraper multi-sources (15+ sites)

    Les sauvegardes PostgreSQL (save_news) sont déléguées à des threads daemon
    APRÈS la sortie du pool principal — le shutdown(wait=True) ne les attend pas.
    """
    parts: list[str] = []
    scraped_to_save = None          # ← capturé hors du with pour l'enregistrer après

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        # Mémoire : analyses récentes du même ticker
        fut_memory = pool.submit(
            memory.get_recent_analyses, ticker, 3, 72
        ) if ticker else None

        # Recherche sémantique dans toutes les analyses passées
        fut_search = pool.submit(memory.search_similar_analyses, question, 2)

        # Données marché temps réel
        fut_macro  = pool.submit(get_macro_summary)
        fut_quote  = pool.submit(get_quote, ticker) if ticker else None
        fut_rss    = pool.submit(get_news_context, ticker)

        # Scraper multi-sources (lancé en parallèle)
        fut_scrape = pool.submit(scrape_all, ticker, question)

        # — Mémoire —
        if fut_memory:
            try:
                past = fut_memory.result(timeout=5)
                mem_text = _format_memory(past)
                if mem_text:
                    parts.append(mem_text)
            except Exception:
                pass

        # — Recherche similaire —
        try:
            similar = fut_search.result(timeout=5)
            if similar:
                lines = ["[Questions similaires analysées]"]
                for s in similar:
                    lines.append(f"• {s['question'][:80]} → {s['answer'][:200]}...")
                parts.append("\n".join(lines))
        except Exception:
            pass

        # — Macro —
        try:
            macro = fut_macro.result(timeout=10)
            if macro:
                parts.append(f"[Macro]\n{macro}")
        except Exception:
            pass

        # — Quote marché —
        if fut_quote:
            try:
                quote = fut_quote.result(timeout=10)
                if quote:
                    parts.append(f"[Marché — {ticker}]\n{quote}")
            except Exception:
                pass

        # — RSS (sources existantes) —
        try:
            rss = fut_rss.result(timeout=10)
            if rss:
                parts.append(f"[Actualités RSS]\n{rss}")
        except Exception:
            pass

        # — Scraper multi-sources —
        try:
            scraped = fut_scrape.result(timeout=12)
            if scraped:
                scraped_to_save = scraped   # ← sera sauvegardé APRÈS la sortie du with
                scraped_text = format_for_context(scraped, max_chars=4000)
                if scraped_text:
                    parts.append(f"[Sources web agrégées]\n{scraped_text}")
        except Exception as e:
            logger.debug(f"scraper timeout: {e}")

    # ↑ Le pool est fermé ici (shutdown wait=True sur les 6 futures de données).
    # save_news est lancé APRÈS pour ne jamais bloquer la réponse HTTP.
    if scraped_to_save and ticker:
        _fire_and_forget(memory.save_news, scraped_to_save, ticker)

    if not parts:
        return "Aucune donnée de marché disponible pour le moment."

    return "\n\n".join(parts)


# ── Point d'entrée ─────────────────────────────────────────────────────

def run(question: str) -> dict:
    """
    Orchestre une question complète.
    Retourne : answer, ticker, lang, data_available.
    """
    lang   = _detect_lang(question)
    ticker = _extract_ticker(question)

    # Construction du contexte enrichi
    context = _build_context(ticker, question, lang)

    # Appel LLM
    answer = ask(question, context)

    # Persistance asynchrone — thread daemon, ne bloque pas la réponse
    _fire_and_forget(
        memory.save_analysis,
        question, answer, ticker, lang, context[:3000],
    )

    return {
        "answer":         answer,
        "ticker":         ticker,
        "lang":           lang,
        "data_available": bool(context),
    }
