from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from hf import analyze_sentiment

app = FastAPI(title="Folio API", version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/api/sentiment")
def sentiment(text: str):
    result = analyze_sentiment(text)
    return {
        "text": text,
        "label": result["label"],
        "score": round(result["score"], 3)
    }
