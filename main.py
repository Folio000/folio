from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from hf import detect_language, translate_to_english, analyze_sentiment

app = FastAPI(title="Folio API", version="3.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

class TextIn(BaseModel):
    text: str

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/api/sentiment")
def sentiment(body: TextIn):
    text = body.text.strip()
    lang = detect_language(text)
    text_en = translate_to_english(text, lang) if lang != "en" else text
    result = analyze_sentiment(text_en)
    return {
        "text": text,
        "label": result["label"],
        "score": round(result["score"], 3),
        "lang": lang
    }
