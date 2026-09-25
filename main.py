"""
main.py — Welto v8 API
Moteur d'éducation financière — FastAPI + Groq + RSS
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from engine.pipeline import run
from data.market import get_snapshot
from engine import memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Welto", version="8.2")
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
    try:
        logger.info(f"[ask] question reçue : {q[:80]}")
        result = run(q)
        logger.info(f"[ask] réponse prête — ticker={result.get('ticker')}, lang={result.get('lang')}")
        return result
    except Exception as e:
        logger.error(f"[ask] ERREUR pipeline : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur pipeline : {e}")


@app.get("/api/snapshot")
def snapshot():
    """Snapshot des principaux indices et titres."""
    assets = get_snapshot()
    return {"assets": assets}


@app.get("/api/suggestions")
def get_suggestions():
    """
    Génère 5 questions éducatives basées sur les actifs les plus actifs du moment.
    Utilise Groq directement (sans le system prompt principal) pour du JSON propre.
    Fallback sur des questions génériques en cas d'erreur.
    """
    import json, re, os
    from groq import Groq

    FALLBACK = [
        "Pourquoi le Nasdaq corrige quand les taux montent ?",
        "Qu'est-ce que le PER et comment l'interpréter ?",
        "Comment l'inflation affecte-t-elle les marchés actions ?",
        "Qu'est-ce qui fait monter ou baisser l'or ?",
        "Que signifient les résultats trimestriels d'Apple ?",
    ]

    try:
        assets = get_snapshot()
        movers = sorted(assets, key=lambda x: abs(x.get("change_pct", 0) or 0), reverse=True)[:6]
        lines = "\n".join(
            f"- {a.get('name', a.get('ticker', ''))}: {a.get('change_pct', 0):+.2f}%"
            for a in movers
        )

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant qui génère des questions financières éducatives. "
                        "Réponds UNIQUEMENT avec un tableau JSON valide. Aucun texte avant ou après."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Actifs les plus actifs aujourd'hui :\n{lines}\n\n"
                        "Génère exactement 5 questions éducatives et pertinentes qu'un investisseur "
                        "particulier pourrait se poser aujourd'hui, basées sur ces données. "
                        "Chaque question : 35 à 70 caractères, en français. "
                        'Format : ["Question 1 ?", "Question 2 ?", "Question 3 ?", "Question 4 ?", "Question 5 ?"]'
                    ),
                },
            ],
            max_tokens=300,
            temperature=0.5,
        )
        raw = completion.choices[0].message.content.strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            questions = json.loads(match.group())
            if isinstance(questions, list) and len(questions) >= 3:
                return {"questions": [str(q) for q in questions[:5]]}
    except Exception as e:
        logger.warning(f"[suggestions] erreur: {e}")

    return {"questions": FALLBACK}


@app.get("/api/health")
def health():
    import os
    stats = memory.get_stats()
    return {
        "status": "ok",
        "version": "8.2",
        "groq_key_present": bool(os.getenv("GROQ_API_KEY")),
        "fred_key_present": bool(os.getenv("FRED_API_KEY")),
        "db_connected": stats.get("db", False),
        "analyses_stored": stats.get("analyses_stored", 0),
        "news_cached": stats.get("news_cached", 0),
    }
