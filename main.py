"""
main.py — Folio v7 API
Moteur d'éducation financière — FastAPI + Groq + RSS
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from engine.pipeline import run
from data.market import get_snapshot
from engine import memory

app = FastAPI(title="Folio", version="8.1")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def on_startup():
    """Initialise les tables PostgreSQL au démarrage."""
    memory.init_db()


# ── Modèles ──────────────────────────────────────────────────────────

class QuestionIn(BaseModel):
    question: str


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/api/ask")
def ask_endpoint(body: QuestionIn):
    """
    Point d'entrée principal.
    Reçoit une question libre, retourne une analyse éducative structurée.
    """
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question vide")
    return run(q)


@app.get("/api/snapshot")
def snapshot():
    """Snapshot des principaux indices et titres."""
    assets = get_snapshot()
    return {"assets": assets}


@app.get("/api/health")
def health():
    import os
    stats = memory.get_stats()
    return {
        "status": "ok",
        "version": "8.1",
        "groq_key_present": bool(os.getenv("GROQ_API_KEY")),
        "fred_key_present": bool(os.getenv("FRED_API_KEY")),
        "db_connected": stats.get("db", False),
        "analyses_stored": stats.get("analyses_stored", 0),
        "news_cached": stats.get("news_cached", 0),
    }
