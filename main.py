from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from hf import detect_language, translate_to_english, analyze_sentiment, extract_ticker, get_market_data

app = FastAPI(title="Folio API", version="4.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

class TextIn(BaseModel):
    text: str

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/api/report")
def report(body: TextIn):
    text = body.text.strip()
    lang = detect_language(text)
    text_en = translate_to_english(text, lang) if lang != "en" else text
    sentiment = analyze_sentiment(text_en)
    ticker = extract_ticker(text)
    market = get_market_data(ticker) if ticker else None
    return {
        "text": text,
        "lang": lang,
        "label": sentiment["label"],
        "score": round(sentiment["score"], 3),
        "ticker": ticker,
        "market": market,
    }
