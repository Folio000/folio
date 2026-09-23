"""
engine/memory.py — Mémoire persistante Folio
PostgreSQL Railway : stockage des analyses + cache articles
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Connexion ─────────────────────────────────────────────────────────

def _get_conn():
    """Connexion PostgreSQL via DATABASE_URL (Railway auto-injecte cette variable)."""
    try:
        import psycopg2
        url = os.getenv("DATABASE_URL", "")
        if not url:
            return None
        conn = psycopg2.connect(url, sslmode="require")
        return conn
    except Exception as e:
        logger.warning(f"Mémoire indisponible : {e}")
        return None


def init_db():
    """Crée les tables si elles n'existent pas encore."""
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id          SERIAL PRIMARY KEY,
                    ticker      TEXT,
                    question    TEXT NOT NULL,
                    lang        TEXT DEFAULT 'fr',
                    context     TEXT,
                    answer      TEXT NOT NULL,
                    keywords    TEXT[],
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_analyses_ticker
                    ON analyses(ticker);
                CREATE INDEX IF NOT EXISTS idx_analyses_created
                    ON analyses(created_at DESC);

                CREATE TABLE IF NOT EXISTS news_cache (
                    id          SERIAL PRIMARY KEY,
                    url         TEXT UNIQUE NOT NULL,
                    ticker      TEXT,
                    source      TEXT,
                    title       TEXT,
                    content     TEXT,
                    published   TIMESTAMPTZ,
                    cached_at   TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_news_ticker
                    ON news_cache(ticker);
                CREATE INDEX IF NOT EXISTS idx_news_cached
                    ON news_cache(cached_at DESC);

                CREATE TABLE IF NOT EXISTS ticker_profiles (
                    ticker      TEXT PRIMARY KEY,
                    name        TEXT,
                    sector      TEXT,
                    description TEXT,
                    info_json   TEXT,
                    updated_at  TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        conn.commit()
        logger.info("DB initialisée")
    except Exception as e:
        logger.error(f"init_db : {e}")
    finally:
        conn.close()


# ── Analyses ──────────────────────────────────────────────────────────

def save_analysis(question: str, answer: str, ticker: Optional[str] = None,
                  lang: str = "fr", context: str = "", keywords: list[str] = None):
    """Persiste une analyse dans PostgreSQL."""
    conn = _get_conn()
    if not conn:
        return
    try:
        kw = keywords or _extract_keywords(question)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO analyses (ticker, question, lang, context, answer, keywords)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (ticker, question, lang, context[:4000], answer, kw))
        conn.commit()
    except Exception as e:
        logger.error(f"save_analysis : {e}")
    finally:
        conn.close()


def get_recent_analyses(ticker: str, limit: int = 3, max_age_hours: int = 48) -> list[dict]:
    """Récupère les analyses récentes pour un ticker — sert de mémoire contextuelle."""
    conn = _get_conn()
    if not conn:
        return []
    try:
        since = datetime.utcnow() - timedelta(hours=max_age_hours)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT question, answer, created_at
                FROM   analyses
                WHERE  ticker = %s AND created_at > %s
                ORDER  BY created_at DESC
                LIMIT  %s
            """, (ticker, since, limit))
            rows = cur.fetchall()
        return [{"question": r[0], "answer": r[1],
                 "at": r[2].strftime("%Y-%m-%d %H:%M")} for r in rows]
    except Exception as e:
        logger.error(f"get_recent_analyses : {e}")
        return []
    finally:
        conn.close()


def search_similar_analyses(question: str, limit: int = 2) -> list[dict]:
    """Recherche par mots-clés dans les analyses passées (full-text simple)."""
    conn = _get_conn()
    if not conn:
        return []
    try:
        kw = _extract_keywords(question)
        if not kw:
            return []
        pattern = " | ".join(kw)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT question, answer, ticker, created_at
                FROM   analyses
                WHERE  to_tsvector('french', question || ' ' || coalesce(answer,''))
                       @@ to_tsquery('french', %s)
                ORDER  BY created_at DESC
                LIMIT  %s
            """, (pattern, limit))
            rows = cur.fetchall()
        return [{"question": r[0], "answer": r[1], "ticker": r[2],
                 "at": r[3].strftime("%Y-%m-%d")} for r in rows]
    except Exception as e:
        logger.warning(f"search_similar : {e}")
        return []
    finally:
        conn.close()


# ── News cache ────────────────────────────────────────────────────────

def get_cached_news(ticker: str, max_age_hours: int = 6) -> list[dict]:
    """Articles déjà scrapés récemment pour ce ticker."""
    conn = _get_conn()
    if not conn:
        return []
    try:
        since = datetime.utcnow() - timedelta(hours=max_age_hours)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT title, content, source, url
                FROM   news_cache
                WHERE  ticker = %s AND cached_at > %s
                ORDER  BY cached_at DESC
                LIMIT  10
            """, (ticker, since))
            rows = cur.fetchall()
        return [{"title": r[0], "content": r[1], "source": r[2], "url": r[3]}
                for r in rows]
    except Exception as e:
        logger.error(f"get_cached_news : {e}")
        return []
    finally:
        conn.close()


def save_news(articles: list[dict], ticker: str):
    """Persiste des articles scrapés dans le cache."""
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            for a in articles:
                url = a.get("url", "")
                if not url:
                    continue
                cur.execute("""
                    INSERT INTO news_cache (url, ticker, source, title, content)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE
                        SET cached_at = NOW(), content = EXCLUDED.content
                """, (url, ticker, a.get("source", ""), a.get("title", ""),
                      (a.get("content", "") or "")[:3000]))
        conn.commit()
    except Exception as e:
        logger.error(f"save_news : {e}")
    finally:
        conn.close()


def save_ticker_profile(ticker: str, info: dict):
    """Cache les infos fondamentales d'un ticker."""
    conn = _get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ticker_profiles (ticker, name, sector, description, info_json, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (ticker) DO UPDATE
                    SET name = EXCLUDED.name, sector = EXCLUDED.sector,
                        description = EXCLUDED.description,
                        info_json = EXCLUDED.info_json, updated_at = NOW()
            """, (ticker, info.get("shortName"), info.get("sector"),
                  info.get("longBusinessSummary", "")[:2000],
                  json.dumps({k: info.get(k) for k in
                              ["marketCap","trailingPE","forwardPE","priceToBook",
                               "revenueGrowth","profitMargins","dividendYield",
                               "52WeekChange","beta"] if info.get(k)})))
        conn.commit()
    except Exception as e:
        logger.error(f"save_ticker_profile : {e}")
    finally:
        conn.close()


def get_ticker_profile(ticker: str, max_age_hours: int = 24) -> Optional[dict]:
    """Récupère le profil fondamental depuis le cache."""
    conn = _get_conn()
    if not conn:
        return None
    try:
        since = datetime.utcnow() - timedelta(hours=max_age_hours)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT name, sector, description, info_json
                FROM   ticker_profiles
                WHERE  ticker = %s AND updated_at > %s
            """, (ticker, since))
            row = cur.fetchone()
        if not row:
            return None
        d = json.loads(row[3]) if row[3] else {}
        d.update({"shortName": row[0], "sector": row[1],
                  "longBusinessSummary": row[2]})
        return d
    except Exception as e:
        logger.error(f"get_ticker_profile : {e}")
        return None
    finally:
        conn.close()


# ── Utils ─────────────────────────────────────────────────────────────

def _extract_keywords(text: str) -> list[str]:
    """Mots significatifs (>3 chars, pas stopwords) pour indexation full-text."""
    STOP = {"que","qui","quoi","est","les","des","une","pour","sur","dans",
            "avec","par","son","ses","leur","leurs","cette","tout","mais",
            "the","and","for","are","was","has","not","this","that","from",
            "what","why","how"}
    words = [w.lower().strip("?!.,;:\"'()") for w in text.split()]
    return list({w for w in words if len(w) > 3 and w not in STOP})[:10]


# ── Stats ─────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Statistiques pour /api/health."""
    conn = _get_conn()
    if not conn:
        return {"db": False}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM analyses")
            analyses_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM news_cache")
            news_count = cur.fetchone()[0]
        return {"db": True, "analyses_stored": analyses_count,
                "news_cached": news_count}
    except Exception as e:
        return {"db": False, "error": str(e)}
    finally:
        conn.close()
