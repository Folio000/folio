cat > ~/folio/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from hf import analyze_sentiment, detect_language

app = FastAPI(title="Folio API", version="2.0")
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
    result = analyze_sentiment(text)
    return {
        "text": text,
        "label": result["label"],
        "score": round(result["score"], 3),
        "lang": lang
    }
EOF
