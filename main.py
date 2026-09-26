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
    Génère 5 questions éducatives dynamiques basées sur les actifs les plus actifs du moment.
    Pas de LLM — génération déterministe par templates pour une fiabilité maximale.
    """
    import random

    FALLBACK = [
        "Pourquoi le Nasdaq corrige quand les taux montent ?",
        "Qu'est-ce que le PER et comment l'interpréter ?",
        "Comment l'inflation affecte-t-elle les marchés actions ?",
        "Qu'est-ce qui fait monter ou baisser l'or ?",
        "Que signifient les résultats trimestriels d'Apple ?",
    ]

    # Templates par type d'actif
    TEMPLATES_UP = [
        "Pourquoi {name} progresse-t-il aujourd'hui ?",
        "Qu'est-ce qui explique la hausse de {name} ?",
        "Quels sont les catalyseurs derrière {name} ?",
        "Comment analyser la progression de {name} ?",
    ]
    TEMPLATES_DOWN = [
        "Pourquoi {name} recule-t-il aujourd'hui ?",
        "Qu'est-ce qui explique la baisse de {name} ?",
        "Faut-il s'inquiéter du repli de {name} ?",
        "Que signifie la correction de {name} ?",
    ]
    GENERIC = [
        "Comment lire un graphique boursier ?",
        "Qu'est-ce que la volatilité des marchés ?",
        "Comment fonctionne un ETF ?",
        "Qu'est-ce que la diversification de portefeuille ?",
        "Comment interpréter les taux d'intérêt de la Fed ?",
        "Qu'est-ce que le PER et comment l'interpréter ?",
        "Comment l'inflation affecte-t-elle les marchés ?",
        "Qu'est-ce qu'une obligation d'État ?",
        "Comment fonctionne le PEA en France ?",
        "Qu'est-ce que la flat tax sur les plus-values ?",
    ]

    try:
        assets = get_snapshot()
        # Top 4 actifs par variation absolue
        movers = sorted(
            [a for a in assets if a.get("name") and a.get("change_pct") is not None],
            key=lambda x: abs(x.get("change_pct", 0) or 0),
            reverse=True,
        )[:4]

        questions = []
        used_generic = random.sample(GENERIC, k=min(3, len(GENERIC)))

        for asset in movers[:3]:
            name = asset.get("name", "").strip()
            if not name:
                continue
            # Raccourcir les noms trop longs
            if len(name) > 20:
                name = name.split(" ")[0]
            pct = asset.get("change_pct", 0) or 0
            tpls = TEMPLATES_UP if pct >= 0 else TEMPLATES_DOWN
            q = random.choice(tpls).format(name=name)
            questions.append(q)

        # Compléter avec des questions génériques
        for gq in used_generic:
            if len(questions) >= 5:
                break
            if gq not in questions:
                questions.append(gq)

        if len(questions) >= 3:
            return {"questions": questions[:5]}

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
