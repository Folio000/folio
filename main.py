from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from hf import (
    detect_language, translate_to_english,
    analyze_sentiment, extract_ticker,
    get_market_data, advise,
    get_universe_snapshot, get_ticker_full,
    get_portfolio_analysis, get_market_news,
)
from data.macro import get_macro_data
from analysis.technicals import get_technicals
from analysis.portfolio import analyze_asset

app = FastAPI(title="Folio API", version="6.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Modèles Pydantic ─────────────────────────────────────────────────

class TextIn(BaseModel):
    text: str


class AdviseIn(BaseModel):
    question: str
    portfolio: Optional[list] = []
    risk_profile: Optional[str] = "modéré"


class PortfolioIn(BaseModel):
    holdings: list  # [{"ticker": "AAPL", "amount": 5000}, ...]


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/")
def home():
    return FileResponse("static/index.html")


# --- Endpoint conseil (principal) ---
@app.post("/api/advise")
def advise_endpoint(body: AdviseIn):
    """
    Conseiller Folio v6 — pipeline complet multilingue.
    Détecte la langue, enrichit avec marché + macro + news + technicals,
    appelle le LLM et répond dans la langue de l'utilisateur.
    """
    return advise(
        question=body.question,
        portfolio=body.portfolio or [],
        risk_profile=body.risk_profile or "modéré",
    )


# --- Legacy endpoint (rétrocompatibilité) ---
@app.post("/api/report")
def report(body: TextIn):
    text = body.text.strip()
    lang = detect_language(text)
    text_en = translate_to_english(text, lang) if lang != "en" else text
    sentiment = analyze_sentiment(text_en)
    ticker = extract_ticker(text)
    market = get_market_data(ticker) if ticker else None
    return {
        "text": text, "lang": lang,
        "label": sentiment["label"], "score": round(sentiment["score"], 3),
        "ticker": ticker, "market": market,
    }


# --- Snapshot marché ---
@app.get("/api/snapshot")
def snapshot():
    data = get_universe_snapshot(max_tickers=18)
    return {"assets": data}


# --- Données d'un ticker (enrichies) ---
@app.get("/api/ticker/{symbol}")
def ticker_data(symbol: str):
    data = get_ticker_full(symbol.upper())
    if not data:
        raise HTTPException(status_code=404, detail="Ticker introuvable")
    return data


# --- Analyse technique d'un ticker ---
@app.get("/api/technicals/{symbol}")
def technicals(symbol: str):
    result = get_technicals(symbol.upper(), period="6mo")
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# --- Données macroéconomiques ---
@app.get("/api/macro")
def macro():
    return get_macro_data()


# --- Analyse de portefeuille ---
@app.post("/api/portfolio/analyze")
def portfolio_analyze(body: PortfolioIn):
    if not body.holdings:
        raise HTTPException(status_code=400, detail="Portefeuille vide")
    return get_portfolio_analysis(body.holdings)


# --- Analyse d'un actif seul (risque/rendement) ---
@app.get("/api/asset/{symbol}")
def asset_analysis(symbol: str):
    from analysis.portfolio import analyze_asset
    result = analyze_asset(symbol.upper())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# --- Actualités marché ---
@app.get("/api/news")
def market_news(lang: str = "fr"):
    articles = get_market_news(user_lang=lang, max_headlines=8)
    return {"articles": articles, "lang": lang}


# --- News + sentiment pour un ticker ---
@app.get("/api/news/{symbol}")
def ticker_news(symbol: str, lang: str = "fr"):
    from data.news import get_ticker_news_sentiment
    from data.fetcher import UNIVERSE
    name = UNIVERSE.get(symbol.upper(), symbol.upper())
    result = get_ticker_news_sentiment(symbol.upper(), name, user_lang=lang)
    return result
