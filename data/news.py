"""
data/news.py — Actualités financières via flux RSS (gratuit, sans clé API)
Sources : Reuters, Yahoo Finance, CNBC, Les Echos
"""

import feedparser
import re
from datetime import datetime, timezone

RSS_FEEDS = [
    # International
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://finance.yahoo.com/rss/topstories",
    # France
    "https://www.lesechos.fr/rss/rss_finance.xml",
    "https://www.latribune.fr/rss/rubriques/economie.html",
]


def _fetch_feed(url: str, max_items: int = 5) -> list[dict]:
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "Folio/7.0"})
        items = []
        for entry in feed.entries[:max_items]:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            # Nettoyer le HTML
            summary = re.sub(r"<[^>]+>", "", summary)[:300]
            published = entry.get("published", "")
            items.append({"title": title, "summary": summary, "published": published})
        return items
    except Exception:
        return []


def _is_relevant(title: str, summary: str, ticker: str | None) -> bool:
    """Filtre basique : l'article parle-t-il du ticker ou d'un thème lié ?"""
    if ticker is None:
        return True
    text = (title + " " + summary).lower()
    # Cherche le ticker ou les mots-clés associés
    aliases = {
        "AAPL": ["apple", "iphone", "mac", "ios"],
        "NVDA": ["nvidia", "gpu", "cuda", "ai chip"],
        "MSFT": ["microsoft", "azure", "openai", "windows"],
        "GOOGL": ["google", "alphabet", "youtube", "android"],
        "AMZN": ["amazon", "aws", "prime"],
        "META": ["meta", "facebook", "instagram", "whatsapp"],
        "TSLA": ["tesla", "musk", "ev", "electric vehicle"],
        "BTC-USD": ["bitcoin", "btc", "crypto"],
        "ETH-USD": ["ethereum", "eth"],
        "GC=F": ["gold", "or", "lingot"],
        "CL=F": ["oil", "pétrole", "crude", "brent", "wti"],
        "^GSPC": ["s&p", "sp500", "wall street"],
        "^IXIC": ["nasdaq", "tech stocks"],
        "^FCHI": ["cac", "bourse de paris", "euronext"],
    }
    keywords = [ticker.lower()] + aliases.get(ticker, [])
    return any(kw in text for kw in keywords)


def get_news_context(ticker: str | None, max_articles: int = 6) -> str:
    """
    Retourne un bloc textuel d'actualités pour injection dans le prompt LLM.
    Si ticker est fourni, filtre les articles pertinents.
    """
    all_articles: list[dict] = []
    for url in RSS_FEEDS[:4]:  # Limite les appels pour la latence
        all_articles.extend(_fetch_feed(url, max_items=8))

    # Filtre si ticker
    if ticker:
        relevant = [
            a for a in all_articles
            if _is_relevant(a["title"], a["summary"], ticker)
        ]
        # Si rien de spécifique, garde les générales
        if not relevant:
            relevant = all_articles
    else:
        relevant = all_articles

    # Déduplication basique sur le titre
    seen: set[str] = set()
    unique: list[dict] = []
    for a in relevant:
        key = a["title"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(a)
        if len(unique) >= max_articles:
            break

    if not unique:
        return ""

    lines = []
    for a in unique:
        lines.append(f"- {a['title']}")
        if a["summary"]:
            lines.append(f"  {a['summary'][:200]}")

    return "\n".join(lines)
