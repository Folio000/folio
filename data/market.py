"""
data/market.py — Données de marché via yfinance
"""

import yfinance as yf


def get_quote(ticker: str) -> str | None:
    """Retourne un résumé textuel du ticker pour injection dans le prompt."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        hist = t.history(period="5d")

        if hist.empty:
            return None

        price = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[-2] if len(hist) >= 2 else price
        change_pct = ((price - prev) / prev * 100) if prev else 0

        name = info.get("longName") or info.get("shortName") or ticker
        currency = info.get("currency", "USD")
        market_cap = info.get("marketCap")
        pe = info.get("trailingPE")
        week52_high = info.get("fiftyTwoWeekHigh")
        week52_low = info.get("fiftyTwoWeekLow")
        analyst_target = info.get("targetMeanPrice")
        recommendation = info.get("recommendationKey", "").replace("_", " ")
        sector = info.get("sector", "")
        volume = hist["Volume"].iloc[-1]

        lines = [
            f"Titre : {name} ({ticker})",
            f"Prix actuel : {price:.2f} {currency} ({change_pct:+.2f}% sur la séance)",
        ]
        if sector:
            lines.append(f"Secteur : {sector}")
        if market_cap:
            lines.append(f"Capitalisation : {market_cap / 1e9:.1f} Md {currency}")
        if pe:
            lines.append(f"PER (TTM) : {pe:.1f}x")
        if week52_high and week52_low:
            lines.append(f"Range 52 semaines : {week52_low:.2f} — {week52_high:.2f} {currency}")
        if analyst_target:
            lines.append(f"Objectif de cours moyen (analystes) : {analyst_target:.2f} {currency}")
        if recommendation:
            lines.append(f"Consensus analystes : {recommendation}")
        if volume:
            lines.append(f"Volume (dernière séance) : {int(volume):,}")

        return "\n".join(lines)
    except Exception:
        return None


def get_snapshot(tickers: list[str] | None = None) -> list[dict]:
    """Retourne une liste de quotes pour un ensemble de tickers."""
    if tickers is None:
        tickers = [
            "^GSPC", "^IXIC", "^FCHI",
            "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
            "GC=F", "CL=F", "BTC-USD",
        ]
    result = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d")
            if hist.empty:
                continue
            price = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2] if len(hist) >= 2 else price
            change_pct = ((price - prev) / prev * 100) if prev else 0
            info = t.info or {}
            result.append({
                "ticker": ticker,
                "name": info.get("shortName") or ticker,
                "price": round(price, 2),
                "change_pct": round(change_pct, 2),
                "currency": info.get("currency", "USD"),
            })
        except Exception:
            continue
    return result
