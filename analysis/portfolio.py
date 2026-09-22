"""
analysis/portfolio.py — Optimisation & analyse de portefeuille
Markowitz (frontière efficiente), Sharpe ratio, VaR (95%), max drawdown,
corrélations, poids optimaux sans librairie externe (numpy uniquement)
"""
import math
import yfinance as yf
from datetime import datetime, timedelta


# ── Utilitaires statistiques ─────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _cov(a: list[float], b: list[float]) -> float:
    """Covariance entre deux séries (même longueur)."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    ma, mb = _mean(a[:n]), _mean(b[:n])
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)


def _corr(a: list[float], b: list[float]) -> float:
    """Corrélation de Pearson."""
    sa, sb = _std(a), _std(b)
    if sa == 0 or sb == 0:
        return 0.0
    return round(_cov(a, b) / (sa * sb), 3)


def _returns(prices: list[float]) -> list[float]:
    """Rendements journaliers logarithmiques."""
    if len(prices) < 2:
        return []
    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]


def _var_historical(returns: list[float], confidence: float = 0.95) -> float:
    """VaR historique (non paramétrique) à confidence% sur 1 jour."""
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    idx = int(len(sorted_r) * (1 - confidence))
    return abs(round(sorted_r[max(idx - 1, 0)], 4))


def _max_drawdown(prices: list[float]) -> float:
    """Max drawdown (perte maximale depuis un sommet)."""
    if len(prices) < 2:
        return 0.0
    peak = prices[0]
    max_dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        dd = (peak - p) / peak
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


def _sharpe(returns: list[float], risk_free_daily: float = 0.0001) -> float:
    """Ratio de Sharpe annualisé (252 jours)."""
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free_daily for r in returns]
    m = _mean(excess)
    s = _std(excess)
    if s == 0:
        return 0.0
    return round(m / s * math.sqrt(252), 3)


# ── Données de marché ────────────────────────────────────────────────

def _fetch_prices(ticker: str, days: int = 252) -> list[float]:
    """Historique de clôture yfinance (n jours)."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{days}d")
        if hist.empty:
            return []
        return [float(v) for v in hist["Close"].tolist()]
    except Exception:
        return []


# ── Analyse d'un actif unique ────────────────────────────────────────

def analyze_asset(ticker: str) -> dict:
    """
    Métriques de risque/rendement pour un actif.
    {
        ticker, annualized_return, volatility, sharpe, var_95,
        max_drawdown, last_price, prices_90d
    }
    """
    prices = _fetch_prices(ticker, days=365)
    if len(prices) < 30:
        return {"ticker": ticker, "error": "Données insuffisantes"}

    rets = _returns(prices)
    ann_return = round(_mean(rets) * 252, 4)   # rendement annualisé
    volatility = round(_std(rets) * math.sqrt(252), 4)  # vol annualisée
    sharpe = _sharpe(rets)
    var_95 = _var_historical(rets, 0.95)
    max_dd = _max_drawdown(prices)

    return {
        "ticker": ticker,
        "last_price": round(prices[-1], 2),
        "annualized_return": ann_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "var_95": var_95,
        "max_drawdown": max_dd,
        "prices_90d": [round(p, 2) for p in prices[-90:]],
    }


# ── Analyse de portefeuille ──────────────────────────────────────────

def analyze_portfolio(holdings: list[dict]) -> dict:
    """
    Analyse d'un portefeuille.
    holdings = [{"ticker": "AAPL", "amount": 5000}, ...]

    Retourne :
    {
        "total_value": float,
        "weights": {ticker: pct},
        "assets": {ticker: {return, vol, sharpe, var, dd}},
        "portfolio_vol": float,
        "portfolio_return": float,
        "portfolio_sharpe": float,
        "portfolio_var_95": float,
        "correlations": {(t1, t2): corr},
        "diversification_score": float,   # 0–1 (1 = très diversifié)
        "recommendations": list[str],
    }
    """
    if not holdings:
        return {"error": "Portefeuille vide"}

    total = sum(h.get("amount", 0) for h in holdings)
    if total <= 0:
        return {"error": "Montants invalides"}

    # Poids
    weights = {h["ticker"]: h.get("amount", 0) / total for h in holdings}

    # Données individuelles
    assets_data = {}
    price_series = {}
    for h in holdings:
        t = h["ticker"]
        data = analyze_asset(t)
        assets_data[t] = data
        if "prices_90d" in data:
            price_series[t] = _returns(data["prices_90d"])

    valid_tickers = [t for t in weights if t in price_series and price_series[t]]

    # Rendement et volatilité du portefeuille (pondérés)
    port_return = 0.0
    port_vol = 0.0

    if valid_tickers:
        # Rendement pondéré
        port_return = sum(
            weights[t] * assets_data[t].get("annualized_return", 0)
            for t in valid_tickers
        )

        # Variance du portefeuille : Σ wi*wj*cov(i,j)
        port_var = 0.0
        for t1 in valid_tickers:
            for t2 in valid_tickers:
                n = min(len(price_series[t1]), len(price_series[t2]))
                cov = _cov(price_series[t1][:n], price_series[t2][:n])
                port_var += weights[t1] * weights[t2] * cov
        port_vol = round(math.sqrt(max(port_var, 0)) * math.sqrt(252), 4)

    port_sharpe = 0.0
    if port_vol > 0:
        risk_free = 0.04  # 4% taux sans risque approximatif
        port_sharpe = round((port_return - risk_free) / port_vol, 3)

    # VaR portefeuille (simulation historique simplifiée)
    if valid_tickers:
        min_len = min(len(price_series[t]) for t in valid_tickers)
        port_daily_rets = []
        for i in range(min_len):
            r = sum(weights[t] * price_series[t][i] for t in valid_tickers)
            port_daily_rets.append(r)
        port_var_95 = _var_historical(port_daily_rets, 0.95)
    else:
        port_var_95 = 0.0

    # Corrélations
    correlations = {}
    for i, t1 in enumerate(valid_tickers):
        for t2 in valid_tickers[i + 1:]:
            n = min(len(price_series[t1]), len(price_series[t2]))
            c = _corr(price_series[t1][:n], price_series[t2][:n])
            correlations[f"{t1}/{t2}"] = c

    # Score de diversification (1 - corrélation moyenne)
    div_score = 1.0
    if correlations:
        avg_corr = _mean(list(correlations.values()))
        div_score = round(max(0.0, 1.0 - abs(avg_corr)), 3)

    # Recommandations
    recommendations = _build_recommendations(
        holdings, assets_data, weights, port_sharpe, div_score, correlations
    )

    return {
        "total_value": round(total, 2),
        "weights": {t: round(w * 100, 1) for t, w in weights.items()},
        "assets": {
            t: {
                "return": assets_data[t].get("annualized_return"),
                "volatility": assets_data[t].get("volatility"),
                "sharpe": assets_data[t].get("sharpe"),
                "var_95": assets_data[t].get("var_95"),
                "max_drawdown": assets_data[t].get("max_drawdown"),
            }
            for t in assets_data if "error" not in assets_data[t]
        },
        "portfolio_return": round(port_return, 4),
        "portfolio_vol": port_vol,
        "portfolio_sharpe": port_sharpe,
        "portfolio_var_95": round(port_var_95, 4),
        "correlations": correlations,
        "diversification_score": div_score,
        "recommendations": recommendations,
    }


def _build_recommendations(
    holdings: list[dict],
    assets: dict,
    weights: dict,
    sharpe: float,
    div_score: float,
    correlations: dict,
) -> list[str]:
    """Génère des recommandations basées sur les métriques du portefeuille."""
    recs = []

    # Concentration excessive
    for ticker, w in weights.items():
        if w > 0.4:
            recs.append(
                f"⚠️ {ticker} représente {w*100:.0f}% du portefeuille — "
                f"concentration élevée, envisager de réduire sous 30%."
            )

    # Faible Sharpe
    if sharpe < 0.5 and sharpe != 0.0:
        recs.append(
            f"📊 Sharpe ratio faible ({sharpe}) — "
            f"le rendement ne compense pas bien le risque pris."
        )
    elif sharpe > 1.5:
        recs.append(f"✅ Excellent Sharpe ratio ({sharpe}) — très bon profil risque/rendement.")

    # Faible diversification
    if div_score < 0.3:
        high_corr = [(k, v) for k, v in correlations.items() if v > 0.8]
        if high_corr:
            pair = high_corr[0][0]
            recs.append(
                f"🔗 Forte corrélation entre {pair} ({high_corr[0][1]}) — "
                f"ces actifs évoluent de concert, la diversification est limitée."
            )

    # Nombre d'actifs
    n = len(holdings)
    if n < 4:
        recs.append(
            f"📌 Portefeuille concentré ({n} actif{'s' if n > 1 else ''}) — "
            f"ajouter 4–8 actifs réduirait le risque non systématique."
        )
    elif n > 20:
        recs.append(
            "📌 Portefeuille très diversifié (>20 actifs) — "
            "le suivi peut devenir complexe ; envisager des ETF."
        )

    if not recs:
        recs.append("✅ Portefeuille équilibré — aucun déséquilibre majeur détecté.")

    return recs


def portfolio_summary_for_prompt(holdings: list[dict], lang: str = "fr") -> str:
    """
    Résumé compact du portefeuille pour injection dans le prompt LLM.
    """
    if not holdings:
        return ""

    analysis = analyze_portfolio(holdings)
    if "error" in analysis:
        return ""

    lines = []
    for t, w in analysis["weights"].items():
        lines.append(f"  {t}: {w}%")

    summary = "\n".join(lines)
    sharpe = analysis.get("portfolio_sharpe", 0)
    vol = analysis.get("portfolio_vol", 0)
    var = analysis.get("portfolio_var_95", 0)

    if lang == "fr":
        return (
            f"Portefeuille ({analysis['total_value']}€) :\n{summary}\n"
            f"Sharpe: {sharpe}, Volatilité: {vol:.1%}, VaR 95%: {var:.1%}/jour"
        )
    elif lang == "de":
        return (
            f"Portfolio ({analysis['total_value']}€) :\n{summary}\n"
            f"Sharpe: {sharpe}, Volatilität: {vol:.1%}, VaR 95%: {var:.1%}/Tag"
        )
    elif lang == "es":
        return (
            f"Cartera ({analysis['total_value']}€) :\n{summary}\n"
            f"Sharpe: {sharpe}, Volatilidad: {vol:.1%}, VaR 95%: {var:.1%}/día"
        )
    elif lang == "it":
        return (
            f"Portafoglio ({analysis['total_value']}€) :\n{summary}\n"
            f"Sharpe: {sharpe}, Volatilità: {vol:.1%}, VaR 95%: {var:.1%}/giorno"
        )
    else:  # en
        return (
            f"Portfolio ({analysis['total_value']}€) :\n{summary}\n"
            f"Sharpe: {sharpe}, Volatility: {vol:.1%}, VaR 95%: {var:.1%}/day"
        )
