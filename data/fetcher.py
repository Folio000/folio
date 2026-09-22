"""
data/fetcher.py — Données de marché enrichies
Sources : yfinance (primaire) + FMP + Alpha Vantage (enrichissement)
"""
import os
import time
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
FMP_KEY = os.getenv("FMP_KEY", "")
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# Univers étendu — CAC40 + S&P500 + Europe + ETF
UNIVERSE = {
    # France — CAC 40
    "AI.PA": "Air Liquide",  "MC.PA": "LVMH",       "TTE.PA": "TotalEnergies",
    "SAN.PA": "Sanofi",      "AIR.PA": "Airbus",     "BNP.PA": "BNP Paribas",
    "KER.PA": "Kering",      "RMS.PA": "Hermès",     "GLE.PA": "Société Générale",
    "OR.PA": "L'Oréal",      "CA.PA": "Carrefour",   "SGO.PA": "Saint-Gobain",
    # USA — Big Tech + Finance
    "AAPL": "Apple",         "NVDA": "NVIDIA",       "MSFT": "Microsoft",
    "AMZN": "Amazon",        "META": "Meta",         "TSLA": "Tesla",
    "GOOGL": "Alphabet",     "JPM": "JPMorgan",      "BRK-B": "Berkshire Hathaway",
    "JNJ": "Johnson & Johnson", "V": "Visa",         "WMT": "Walmart",
    "XOM": "ExxonMobil",     "BAC": "Bank of America",
    # Europe
    "SIE.DE": "Siemens",     "SAP.DE": "SAP",        "BMW.DE": "BMW",
    "ALV.DE": "Allianz",     "ASML.AS": "ASML",      "NESN.SW": "Nestlé",
    "ROG.SW": "Roche",       "ITX.MC": "Inditex",    "SHEL.L": "Shell",
    "AZN.L": "AstraZeneca",  "NOVN.SW": "Novartis",
    # ETF
    "CW8.PA": "Amundi MSCI World", "ESE.PA": "iShares S&P 500",
    "IWDA.AS": "iShares Core MSCI World",
    # Crypto trackers (ETF)
    "IBIT": "iShares Bitcoin Trust",
}

# Nom → ticker (extraction dans texte libre)
TICKER_MAP = {
    "apple": "AAPL", "nvidia": "NVDA", "microsoft": "MSFT",
    "amazon": "AMZN", "tesla": "TSLA", "meta": "META",
    "alphabet": "GOOGL", "google": "GOOGL", "netflix": "NFLX",
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "berkshire": "BRK-B", "visa": "V", "walmart": "WMT",
    "exxon": "XOM", "johnson": "JNJ",
    "lvmh": "MC.PA", "air liquide": "AI.PA", "totalenergies": "TTE.PA",
    "total": "TTE.PA", "bnp": "BNP.PA", "bnp paribas": "BNP.PA",
    "sanofi": "SAN.PA", "airbus": "AIR.PA", "kering": "KER.PA",
    "hermes": "RMS.PA", "hermès": "RMS.PA", "loreal": "OR.PA",
    "l'oréal": "OR.PA", "société générale": "GLE.PA", "carrefour": "CA.PA",
    "saint gobain": "SGO.PA", "saint-gobain": "SGO.PA",
    "volkswagen": "VOW.DE", "vw": "VOW.DE", "deutsche bank": "DBK.DE",
    "siemens": "SIE.DE", "bmw": "BMW.DE", "mercedes": "MBG.DE",
    "bayer": "BAYN.DE", "sap": "SAP.DE", "allianz": "ALV.DE",
    "asml": "ASML.AS", "nestle": "NESN.SW", "nestlé": "NESN.SW",
    "roche": "ROG.SW", "novartis": "NOVN.SW",
    "inditex": "ITX.MC", "zara": "ITX.MC",
    "santander": "SAN.MC", "bbva": "BBVA.MC",
    "shell": "SHEL.L", "astrazeneca": "AZN.L",
    "amundi world": "CW8.PA", "msci world": "CW8.PA",
    "sp500": "ESE.PA", "s&p 500": "ESE.PA", "s&p500": "ESE.PA",
    "bitcoin": "IBIT", "btc": "IBIT",
}


def _fmp_profile(ticker: str) -> dict:
    """FMP : profil entreprise (description, CEO, secteur, notation ESG)."""
    if not FMP_KEY:
        return {}
    # Convertir tickers européens (MC.PA → MC) pour FMP
    symbol = ticker.split(".")[0]
    try:
        r = requests.get(
            f"https://financialmodelingprep.com/api/v3/profile/{symbol}",
            params={"apikey": FMP_KEY},
            timeout=8
        )
        data = r.json()
        if isinstance(data, list) and data:
            p = data[0]
            return {
                "description": p.get("description", "")[:300],
                "ceo": p.get("ceo", ""),
                "beta": p.get("beta"),
                "dcf_diff": p.get("dcfDiff"),  # écart cours / valeur intrinsèque
                "rating": None,
            }
    except Exception:
        pass
    return {}


def _fmp_rating(ticker: str) -> str | None:
    """FMP : notation financière agrégée (A+, B-, etc.)."""
    if not FMP_KEY:
        return None
    symbol = ticker.split(".")[0]
    try:
        r = requests.get(
            f"https://financialmodelingprep.com/api/v3/rating/{symbol}",
            params={"apikey": FMP_KEY},
            timeout=8
        )
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get("rating")
    except Exception:
        pass
    return None


def _alpha_rsi(ticker: str) -> float | None:
    """Alpha Vantage : RSI(14) — couche supplémentaire si TA échoue."""
    if not ALPHA_KEY:
        return None
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "RSI",
                "symbol": ticker.split(".")[0],
                "interval": "daily",
                "time_period": 14,
                "series_type": "close",
                "apikey": ALPHA_KEY,
            },
            timeout=10
        )
        data = r.json()
        values = data.get("Technical Analysis: RSI", {})
        if values:
            latest_date = sorted(values.keys(), reverse=True)[0]
            return float(values[latest_date]["RSI"])
    except Exception:
        pass
    return None


def get_market_data(ticker: str, enrich: bool = False) -> dict | None:
    """
    Données de marché pour un ticker.
    enrich=True : ajoute FMP profile + rating (plus lent, pour les appels ciblés).
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            return None

        last_price = float(hist["Close"].iloc[-1])
        prev_price = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_price
        change_pct = ((last_price - prev_price) / prev_price) * 100 if prev_price else 0

        full = {}
        try:
            full = t.info
        except Exception:
            pass

        result = {
            "ticker": ticker,
            "name": full.get("shortName") or full.get("longName") or UNIVERSE.get(ticker, ticker),
            "price": round(last_price, 2),
            "change_pct": round(change_pct, 2),
            "currency": full.get("currency", "EUR"),
            "market_cap": full.get("marketCap"),
            "pe_ratio": full.get("trailingPE"),
            "forward_pe": full.get("forwardPE"),
            "peg_ratio": full.get("pegRatio"),
            "dividend_yield": full.get("dividendYield"),
            "week52_high": full.get("fiftyTwoWeekHigh"),
            "week52_low": full.get("fiftyTwoWeekLow"),
            "volume": full.get("volume"),
            "avg_volume": full.get("averageVolume"),
            "sector": full.get("sector"),
            "industry": full.get("industry"),
            "country": full.get("country"),
            "analyst_target": full.get("targetMeanPrice"),
            "recommendation": full.get("recommendationKey"),  # buy/hold/sell
            "esg_score": full.get("totalEsg"),
            # Données historiques pour graphes
            "hist_closes": [round(float(v), 2) for v in hist["Close"].tolist()[-5:]],
            "hist_dates": [str(d.date()) for d in hist.index.tolist()[-5:]],
        }

        # Enrichissement FMP (optionnel)
        if enrich:
            profile = _fmp_profile(ticker)
            result.update({
                "description": profile.get("description", ""),
                "ceo": profile.get("ceo", ""),
                "beta": profile.get("beta"),
                "dcf_diff": profile.get("dcf_diff"),
                "rating": _fmp_rating(ticker),
            })

        return result

    except Exception:
        return None


def get_universe_snapshot(max_tickers: int = 16) -> list[dict]:
    """Snapshot parallèle des tickers de l'univers d'investissement."""
    tickers = list(UNIVERSE.keys())[:max_tickers]
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_market_data, t, False): t for t in tickers}
        for future in as_completed(futures):
            data = future.result()
            if data:
                results.append(data)
    results.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    return results


def get_ticker_full(ticker: str) -> dict | None:
    """Données complètes d'un ticker unique (avec enrichissement FMP)."""
    return get_market_data(ticker.upper(), enrich=True)


def get_news_headlines(tickers: list[str]) -> list[str]:
    """Titres Yahoo Finance pour les tickers donnés."""
    headlines = []
    seen = set()
    for ticker in tickers[:5]:
        try:
            t = yf.Ticker(ticker)
            for article in (t.news or [])[:3]:
                title = article.get("title", "")
                if title and title not in seen:
                    seen.add(title)
                    headlines.append(title)
        except Exception:
            pass
    return headlines[:10]


def extract_ticker(text: str) -> str | None:
    """Extrait un ticker d'un texte libre ($ notation ou noms communs)."""
    import re
    match = re.search(r'\$([A-Z]{1,5}(?:\.[A-Z]{1,2})?)', text)
    if match:
        return match.group(1)
    text_lower = text.lower()
    for name, ticker in sorted(TICKER_MAP.items(), key=lambda x: -len(x[0])):
        if name in text_lower:
            return ticker
    return None
