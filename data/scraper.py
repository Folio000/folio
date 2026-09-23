"""
data/scraper.py — Scraper multi-sources Folio
Agrège actualités et fondamentaux depuis ~15 sources FR + EN + officielles.
Robuste : chaque source est isolée, une panne n'en affecte pas d'autres.
"""

import re
import time
import logging
import feedparser
import requests
from typing import Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Session HTTP commune ───────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})
TIMEOUT = 8


def _get(url: str, **kw) -> Optional[requests.Response]:
    try:
        r = SESSION.get(url, timeout=TIMEOUT, **kw)
        r.raise_for_status()
        return r
    except Exception as e:
        logger.debug(f"GET {url} — {e}")
        return None


def _soup(url: str, **kw) -> Optional[BeautifulSoup]:
    r = _get(url, **kw)
    if r:
        return BeautifulSoup(r.text, "lxml")
    return None


def _text(tag) -> str:
    return tag.get_text(" ", strip=True) if tag else ""


# ═══════════════════════════════════════════════════════════════════════
# SOURCES EN (English)
# ═══════════════════════════════════════════════════════════════════════

def scrape_yahoo_finance(ticker: str) -> list[dict]:
    """Yahoo Finance — actualités par ticker."""
    articles = []
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        soup = _soup(url)
        if not soup:
            return []
        for item in soup.select("li.js-stream-content, div[data-testid='news-stream'] li")[:8]:
            title_tag = item.select_one("h3, h4, a[data-ylk]")
            link_tag  = item.select_one("a[href]")
            para_tag  = item.select_one("p")
            title = _text(title_tag)
            href  = link_tag["href"] if link_tag else ""
            if href and not href.startswith("http"):
                href = "https://finance.yahoo.com" + href
            if title:
                articles.append({
                    "source": "Yahoo Finance",
                    "title": title,
                    "content": _text(para_tag),
                    "url": href,
                })
    except Exception as e:
        logger.debug(f"yahoo_finance({ticker}): {e}")
    return articles


def scrape_marketwatch(ticker: str) -> list[dict]:
    """MarketWatch — dernières actualités."""
    articles = []
    try:
        url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"
        soup = _soup(url)
        if not soup:
            return []
        for item in soup.select("div.article__content, div.collection__elements article")[:6]:
            title_tag = item.select_one("h3, h4, .article__headline")
            link_tag  = item.select_one("a[href]")
            para_tag  = item.select_one("p")
            title = _text(title_tag)
            href  = link_tag["href"] if link_tag else ""
            if title:
                articles.append({
                    "source": "MarketWatch",
                    "title": title,
                    "content": _text(para_tag),
                    "url": href,
                })
    except Exception as e:
        logger.debug(f"marketwatch({ticker}): {e}")
    return articles


def scrape_seeking_alpha(ticker: str) -> list[dict]:
    """Seeking Alpha — RSS feed (articles gratuits)."""
    articles = []
    try:
        feed = feedparser.parse(
            f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
        )
        for entry in feed.entries[:5]:
            articles.append({
                "source": "Seeking Alpha",
                "title": entry.get("title", ""),
                "content": BeautifulSoup(
                    entry.get("summary", ""), "lxml"
                ).get_text(" ", strip=True)[:800],
                "url": entry.get("link", ""),
            })
    except Exception as e:
        logger.debug(f"seeking_alpha({ticker}): {e}")
    return articles


def scrape_stockanalysis(ticker: str) -> list[dict]:
    """StockAnalysis — données fondamentales et résumé."""
    articles = []
    try:
        url = f"https://stockanalysis.com/stocks/{ticker.lower()}/"
        soup = _soup(url)
        if not soup:
            return []
        # Statistiques clés
        stats = {}
        for row in soup.select("table tr, div[data-testid='overview-info'] div"):
            cells = row.find_all(["td", "th"])
            if len(cells) == 2:
                k = _text(cells[0])
                v = _text(cells[1])
                if k and v:
                    stats[k] = v

        desc_tag = soup.select_one("div.mt-5.text-gray-600, p.description, div[class*='description']")
        desc = _text(desc_tag)[:1000] if desc_tag else ""

        content_parts = []
        if stats:
            content_parts.append("Fondamentaux : " + " | ".join(
                f"{k}: {v}" for k, v in list(stats.items())[:15]
            ))
        if desc:
            content_parts.append(desc)

        if content_parts:
            articles.append({
                "source": "StockAnalysis",
                "title": f"Analyse fondamentale {ticker}",
                "content": "\n".join(content_parts),
                "url": url,
            })
    except Exception as e:
        logger.debug(f"stockanalysis({ticker}): {e}")
    return articles


def scrape_reuters_search(query: str) -> list[dict]:
    """Reuters — RSS search."""
    articles = []
    try:
        feed = feedparser.parse(
            f"https://feeds.reuters.com/reuters/businessNews"
        )
        q_lower = query.lower()
        for entry in feed.entries[:20]:
            title = entry.get("title", "")
            if q_lower in title.lower() or any(
                w in title.lower() for w in q_lower.split()[:3]
            ):
                articles.append({
                    "source": "Reuters",
                    "title": title,
                    "content": BeautifulSoup(
                        entry.get("summary", ""), "lxml"
                    ).get_text(" ", strip=True)[:600],
                    "url": entry.get("link", ""),
                })
    except Exception as e:
        logger.debug(f"reuters_search: {e}")
    return articles[:4]


def scrape_morningstar(ticker: str) -> list[dict]:
    """Morningstar — résumé analytique (données publiques)."""
    articles = []
    try:
        url = f"https://www.morningstar.com/stocks/xnas/{ticker.lower()}/quote"
        soup = _soup(url, headers={"Referer": "https://www.morningstar.com/"})
        if not soup:
            return []
        for sel in ["div[class*='quote-summary']", "div[class*='KeyStats']",
                    "section[class*='analysis']"]:
            block = soup.select_one(sel)
            if block:
                text = _text(block)[:1000]
                if text:
                    articles.append({
                        "source": "Morningstar",
                        "title": f"Morningstar — {ticker}",
                        "content": text,
                        "url": url,
                    })
                    break
    except Exception as e:
        logger.debug(f"morningstar({ticker}): {e}")
    return articles


# ═══════════════════════════════════════════════════════════════════════
# SOURCES FR (French)
# ═══════════════════════════════════════════════════════════════════════

def scrape_boursorama(ticker: str) -> list[dict]:
    """Boursorama — actualités valeur (marché FR prioritaire)."""
    articles = []
    try:
        # Boursorama utilise des codes internes mais Yahoo ticker fonctionne pour news
        url = f"https://www.boursorama.com/recherche/actu/?query={ticker}"
        soup = _soup(url)
        if not soup:
            return []
        for item in soup.select("article, div.c-news-item")[:6]:
            title_tag = item.select_one("h2, h3, .c-news-item__title")
            link_tag  = item.select_one("a[href]")
            para_tag  = item.select_one("p, .c-news-item__intro")
            title = _text(title_tag)
            href  = link_tag["href"] if link_tag else ""
            if href and href.startswith("/"):
                href = "https://www.boursorama.com" + href
            if title:
                articles.append({
                    "source": "Boursorama",
                    "title": title,
                    "content": _text(para_tag),
                    "url": href,
                })
    except Exception as e:
        logger.debug(f"boursorama({ticker}): {e}")
    return articles


def scrape_zonebourse(ticker: str) -> list[dict]:
    """Zonebourse — actualités et analyse technique."""
    articles = []
    try:
        url = f"https://www.zonebourse.com/recherche/?q={ticker}&type=news"
        soup = _soup(url)
        if not soup:
            return []
        for item in soup.select("article, .news-item, div[class*='article']")[:5]:
            title_tag = item.select_one("h2, h3, .article-title")
            link_tag  = item.select_one("a[href]")
            para_tag  = item.select_one("p")
            title = _text(title_tag)
            href  = link_tag["href"] if link_tag else ""
            if href and href.startswith("/"):
                href = "https://www.zonebourse.com" + href
            if title:
                articles.append({
                    "source": "Zonebourse",
                    "title": title,
                    "content": _text(para_tag),
                    "url": href,
                })
    except Exception as e:
        logger.debug(f"zonebourse({ticker}): {e}")
    return articles


def scrape_bfm_bourse() -> list[dict]:
    """BFM Bourse — fil d'actualité général marché."""
    articles = []
    try:
        feed = feedparser.parse("https://bfmbusiness.bfmtv.com/rss/bfm-bourse.xml")
        for entry in feed.entries[:6]:
            articles.append({
                "source": "BFM Bourse",
                "title": entry.get("title", ""),
                "content": BeautifulSoup(
                    entry.get("summary", ""), "lxml"
                ).get_text(" ", strip=True)[:600],
                "url": entry.get("link", ""),
            })
    except Exception as e:
        logger.debug(f"bfm_bourse: {e}")
    return articles


def scrape_les_echos(query: str) -> list[dict]:
    """Les Echos — RSS marchés (articles filtrés par query)."""
    articles = []
    try:
        feed = feedparser.parse("https://www.lesechos.fr/rss/rss_finance-marches.xml")
        q_lower = query.lower()
        for entry in feed.entries[:20]:
            title = entry.get("title", "")
            if any(w in title.lower() for w in q_lower.split()[:3]):
                articles.append({
                    "source": "Les Echos",
                    "title": title,
                    "content": BeautifulSoup(
                        entry.get("summary", ""), "lxml"
                    ).get_text(" ", strip=True)[:600],
                    "url": entry.get("link", ""),
                })
    except Exception as e:
        logger.debug(f"les_echos: {e}")
    return articles[:4]


def scrape_investir_les_echos(ticker: str) -> list[dict]:
    """Investir.lesechos.fr — fiche valeur."""
    articles = []
    try:
        url = f"https://investir.lesechos.fr/actions/actualites/{ticker.lower()}"
        soup = _soup(url)
        if not soup:
            return []
        for item in soup.select("article, .news-item")[:5]:
            title_tag = item.select_one("h2, h3")
            link_tag  = item.select_one("a[href]")
            para_tag  = item.select_one("p")
            title = _text(title_tag)
            href  = link_tag["href"] if link_tag else ""
            if title:
                articles.append({
                    "source": "Investir (Les Echos)",
                    "title": title,
                    "content": _text(para_tag),
                    "url": href,
                })
    except Exception as e:
        logger.debug(f"investir_echos({ticker}): {e}")
    return articles


# ═══════════════════════════════════════════════════════════════════════
# SOURCES OFFICIELLES
# ═══════════════════════════════════════════════════════════════════════

def scrape_sec_edgar(ticker: str) -> list[dict]:
    """SEC EDGAR — derniers filings (8-K, 10-Q) via API JSON gratuite."""
    articles = []
    try:
        # Cherche le CIK du ticker
        r = _get(f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom"
                 f"&startdt=2024-01-01&forms=8-K,10-Q&hits.hits._source=file_date,entity_name,file_num")
        if not r:
            return []
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])[:3]
        for h in hits:
            src = h.get("_source", {})
            articles.append({
                "source": "SEC EDGAR",
                "title": f"Filing {src.get('form_type','?')} — {src.get('entity_name',ticker)}",
                "content": f"Date : {src.get('file_date','')} | Form : {src.get('form_type','')}",
                "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=8-K&dateb=&owner=include&count=10",
            })
    except Exception as e:
        logger.debug(f"sec_edgar({ticker}): {e}")
    return articles


def scrape_cnbc_rss(query: str) -> list[dict]:
    """CNBC — RSS finance (filtré par query)."""
    articles = []
    try:
        feed = feedparser.parse("https://www.cnbc.com/id/10000664/device/rss/rss.html")
        q_lower = query.lower()
        for entry in feed.entries[:20]:
            title = entry.get("title", "")
            if any(w in title.lower() for w in q_lower.split()[:3]):
                articles.append({
                    "source": "CNBC",
                    "title": title,
                    "content": BeautifulSoup(
                        entry.get("summary", ""), "lxml"
                    ).get_text(" ", strip=True)[:600],
                    "url": entry.get("link", ""),
                })
    except Exception as e:
        logger.debug(f"cnbc: {e}")
    return articles[:4]


def scrape_ft_rss() -> list[dict]:
    """Financial Times — RSS free headlines."""
    articles = []
    try:
        feed = feedparser.parse("https://www.ft.com/?format=rss")
        for entry in feed.entries[:5]:
            articles.append({
                "source": "Financial Times",
                "title": entry.get("title", ""),
                "content": entry.get("summary", "")[:400],
                "url": entry.get("link", ""),
            })
    except Exception as e:
        logger.debug(f"ft_rss: {e}")
    return articles


def scrape_wsj_rss() -> list[dict]:
    """Wall Street Journal — RSS headlines."""
    articles = []
    try:
        feed = feedparser.parse("https://feeds.a.dj.com/rss/RSSMarketsMain.xml")
        for entry in feed.entries[:5]:
            articles.append({
                "source": "WSJ Markets",
                "title": entry.get("title", ""),
                "content": entry.get("summary", "")[:400],
                "url": entry.get("link", ""),
            })
    except Exception as e:
        logger.debug(f"wsj_rss: {e}")
    return articles


# ═══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

def scrape_all(ticker: Optional[str], question: str) -> list[dict]:
    """
    Lance tous les scrapers en parallèle.
    Retourne une liste consolidée d'articles déduplicatés.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = {}

    if ticker:
        tasks["yahoo"]        = lambda: scrape_yahoo_finance(ticker)
        tasks["marketwatch"]  = lambda: scrape_marketwatch(ticker)
        tasks["seeking"]      = lambda: scrape_seeking_alpha(ticker)
        tasks["stockanalysis"]= lambda: scrape_stockanalysis(ticker)
        tasks["morningstar"]  = lambda: scrape_morningstar(ticker)
        tasks["boursorama"]   = lambda: scrape_boursorama(ticker)
        tasks["zonebourse"]   = lambda: scrape_zonebourse(ticker)
        tasks["investir"]     = lambda: scrape_investir_les_echos(ticker)
        tasks["sec"]          = lambda: scrape_sec_edgar(ticker)

    # Sources générales (toujours actives)
    tasks["bfm"]     = lambda: scrape_bfm_bourse()
    tasks["ft"]      = lambda: scrape_ft_rss()
    tasks["wsj"]     = lambda: scrape_wsj_rss()
    tasks["reuters"] = lambda: scrape_reuters_search(question)
    tasks["echos"]   = lambda: scrape_les_echos(question)
    tasks["cnbc"]    = lambda: scrape_cnbc_rss(question)

    results = []
    seen_titles: set[str] = set()

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(futures, timeout=10):
            try:
                for art in future.result():
                    t = art.get("title", "").strip()
                    if t and t not in seen_titles and len(t) > 10:
                        seen_titles.add(t)
                        results.append(art)
            except Exception:
                pass

    logger.info(f"Scraper : {len(results)} articles collectés pour '{ticker or question}'")
    return results


def format_for_context(articles: list[dict], max_chars: int = 4000) -> str:
    """Formate les articles en bloc texte pour le contexte LLM."""
    if not articles:
        return ""
    lines = []
    total = 0
    for a in articles:
        source  = a.get("source", "")
        title   = a.get("title", "")
        content = (a.get("content") or "").strip()
        block   = f"[{source}] {title}"
        if content:
            block += f"\n{content[:300]}"
        block += "\n"
        if total + len(block) > max_chars:
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines)
