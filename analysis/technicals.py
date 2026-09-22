"""
analysis/technicals.py — Indicateurs techniques
RSI, MACD, Bollinger Bands, EMA 20/50/200, signaux BUY/HOLD/SELL
Calcul sur historique yfinance (sans dépendance TA-lib pour simplicité)
"""
import math
import yfinance as yf


def _ema(prices: list[float], period: int) -> list[float]:
    """Calcul EMA sur une liste de prix."""
    if len(prices) < period:
        return []
    k = 2 / (period + 1)
    ema = [sum(prices[:period]) / period]
    for p in prices[period:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return ema


def _rsi(prices: list[float], period: int = 14) -> float | None:
    """RSI de Wilder."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def _macd(prices: list[float]) -> dict | None:
    """MACD (12,26,9) : ligne MACD, signal, histogramme."""
    if len(prices) < 35:
        return None
    ema12 = _ema(prices, 12)
    ema26 = _ema(prices, 26)
    # Aligner : ema26 est plus court
    offset = len(ema12) - len(ema26)
    macd_line = [ema12[i + offset] - ema26[i] for i in range(len(ema26))]
    if len(macd_line) < 9:
        return None
    signal_line = _ema(macd_line, 9)
    hist = macd_line[-1] - signal_line[-1] if signal_line else 0
    return {
        "macd": round(macd_line[-1], 4),
        "signal": round(signal_line[-1], 4),
        "histogram": round(hist, 4),
    }


def _bollinger(prices: list[float], period: int = 20, std_mult: float = 2.0) -> dict | None:
    """Bandes de Bollinger (20, ±2σ)."""
    if len(prices) < period:
        return None
    window = prices[-period:]
    mean = sum(window) / period
    variance = sum((p - mean) ** 2 for p in window) / period
    std = math.sqrt(variance)
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    last = prices[-1]
    # Position relative dans les bandes (0 = lower, 1 = upper)
    width = upper - lower
    pct_b = (last - lower) / width if width > 0 else 0.5
    return {
        "upper": round(upper, 2),
        "middle": round(mean, 2),
        "lower": round(lower, 2),
        "pct_b": round(pct_b, 3),
    }


def _signal_rsi(rsi: float | None) -> str:
    if rsi is None:
        return "HOLD"
    if rsi < 30:
        return "BUY"   # Survente
    if rsi > 70:
        return "SELL"  # Surachat
    return "HOLD"


def _signal_macd(macd: dict | None) -> str:
    if not macd:
        return "HOLD"
    if macd["histogram"] > 0 and macd["macd"] > macd["signal"]:
        return "BUY"
    if macd["histogram"] < 0 and macd["macd"] < macd["signal"]:
        return "SELL"
    return "HOLD"


def _signal_bollinger(bb: dict | None, current_price: float) -> str:
    if not bb:
        return "HOLD"
    if current_price <= bb["lower"]:
        return "BUY"
    if current_price >= bb["upper"]:
        return "SELL"
    return "HOLD"


def _signal_ema(price: float, ema20: float | None, ema50: float | None) -> str:
    if ema20 is None or ema50 is None:
        return "HOLD"
    if price > ema20 > ema50:
        return "BUY"   # Tendance haussière
    if price < ema20 < ema50:
        return "SELL"  # Tendance baissière
    return "HOLD"


def get_technicals(ticker: str, period: str = "6mo") -> dict:
    """
    Calcule les indicateurs techniques pour un ticker.
    Retourne :
    {
        "rsi": float,
        "macd": dict,
        "bollinger": dict,
        "ema20": float,
        "ema50": float,
        "ema200": float | None,
        "signals": { rsi, macd, bb, ema, overall },
        "signal_score": int,  # -3 à +3 (négatif = bearish, positif = bullish)
        "trend": "bullish|bearish|neutral",
    }
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty or len(hist) < 20:
            return {"error": "Données insuffisantes"}

        prices = [float(v) for v in hist["Close"].tolist()]
        current = prices[-1]

        rsi = _rsi(prices, 14)
        macd = _macd(prices)
        bb = _bollinger(prices, 20)

        ema20_series = _ema(prices, 20)
        ema50_series = _ema(prices, 50) if len(prices) >= 50 else []
        ema200_series = _ema(prices, 200) if len(prices) >= 200 else []

        ema20 = round(ema20_series[-1], 2) if ema20_series else None
        ema50 = round(ema50_series[-1], 2) if ema50_series else None
        ema200 = round(ema200_series[-1], 2) if ema200_series else None

        sig_rsi = _signal_rsi(rsi)
        sig_macd = _signal_macd(macd)
        sig_bb = _signal_bollinger(bb, current)
        sig_ema = _signal_ema(current, ema20, ema50)

        # Score : BUY=+1, SELL=-1, HOLD=0
        score_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        score = (
            score_map[sig_rsi]
            + score_map[sig_macd]
            + score_map[sig_bb]
            + score_map[sig_ema]
        )

        if score >= 2:
            overall = "BUY"
            trend = "bullish"
        elif score <= -2:
            overall = "SELL"
            trend = "bearish"
        else:
            overall = "HOLD"
            trend = "neutral"

        return {
            "ticker": ticker,
            "current_price": round(current, 2),
            "rsi": rsi,
            "macd": macd,
            "bollinger": bb,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "signals": {
                "rsi": sig_rsi,
                "macd": sig_macd,
                "bollinger": sig_bb,
                "ema": sig_ema,
                "overall": overall,
            },
            "signal_score": score,
            "trend": trend,
        }

    except Exception as e:
        return {"error": str(e)}


def format_signal(tech: dict, lang: str = "fr") -> str:
    """
    Formate les signaux techniques en texte court (pour le prompt LLM).
    Multilingue : fr / en / de / es / it
    """
    if "error" in tech:
        return ""

    ticker = tech.get("ticker", "")
    rsi = tech.get("rsi")
    trend = tech.get("trend", "neutral")
    overall = tech.get("signals", {}).get("overall", "HOLD")
    score = tech.get("signal_score", 0)

    labels = {
        "fr": {
            "bullish": "haussier", "bearish": "baissier", "neutral": "neutre",
            "BUY": "ACHETER", "SELL": "VENDRE", "HOLD": "CONSERVER",
            "prefix": "Analyse technique",
        },
        "en": {
            "bullish": "bullish", "bearish": "bearish", "neutral": "neutral",
            "BUY": "BUY", "SELL": "SELL", "HOLD": "HOLD",
            "prefix": "Technical analysis",
        },
        "de": {
            "bullish": "bullisch", "bearish": "bärisch", "neutral": "neutral",
            "BUY": "KAUFEN", "SELL": "VERKAUFEN", "HOLD": "HALTEN",
            "prefix": "Technische Analyse",
        },
        "es": {
            "bullish": "alcista", "bearish": "bajista", "neutral": "neutral",
            "BUY": "COMPRAR", "SELL": "VENDER", "HOLD": "MANTENER",
            "prefix": "Análisis técnico",
        },
        "it": {
            "bullish": "rialzista", "bearish": "ribassista", "neutral": "neutro",
            "BUY": "COMPRARE", "SELL": "VENDERE", "HOLD": "MANTENERE",
            "prefix": "Analisi tecnica",
        },
    }
    l = labels.get(lang, labels["en"])

    rsi_str = f"RSI={rsi}" if rsi else ""
    trend_str = l.get(trend, trend)
    signal_str = l.get(overall, overall)

    return (
        f"{l['prefix']} {ticker}: {signal_str} "
        f"(tendance {trend_str}, score {score:+d}, {rsi_str})"
    )
