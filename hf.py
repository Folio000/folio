"""
hf.py — Orchestrateur central Folio v6
Pipeline complet multilingue avec conseiller rule-based + LLM optionnel
Priorité : budget respecté + profil de risque adapté + langue de l'utilisateur
"""
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from data.fetcher import (
    get_universe_snapshot, get_ticker_full,
    get_news_headlines, extract_ticker,
    UNIVERSE, TICKER_MAP,
)
from data.news import (
    detect_language, translate_to_english,
    analyze_sentiment, get_ticker_news_sentiment,
    get_market_news, analyze_monetary_policy,
)
from data.macro import get_macro_summary
from analysis.technicals import get_technicals, format_signal
from analysis.portfolio import analyze_portfolio, portfolio_summary_for_prompt

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

__all__ = [
    "detect_language", "translate_to_english", "analyze_sentiment",
    "extract_ticker", "get_market_data", "get_universe_snapshot",
    "advise", "get_ticker_full", "get_portfolio_analysis",
    "get_market_news", "UNIVERSE", "TICKER_MAP",
]

# ── Catégories d'actifs par profil de risque ─────────────────────────
RISK_UNIVERSE = {
    "prudent": {
        "tickers": ["CW8.PA", "ESE.PA", "IWDA.AS", "OR.PA", "NESN.SW", "AI.PA", "SAN.PA", "BNP.PA"],
        "description": {
            "fr": "ETF diversifiés et valeurs défensives (dividendes stables, faible volatilité)",
            "en": "Diversified ETFs and defensive stocks (stable dividends, low volatility)",
            "de": "Diversifizierte ETFs und defensive Aktien (stabile Dividenden, geringe Volatilität)",
            "es": "ETFs diversificados y valores defensivos (dividendos estables, baja volatilidad)",
            "it": "ETF diversificati e titoli difensivi (dividendi stabili, bassa volatilità)",
        },
        "allocation": {"etf": 0.70, "blue_chip": 0.30},
    },
    "modéré": {
        "tickers": ["CW8.PA", "MSFT", "AAPL", "MC.PA", "OR.PA", "ASML.AS", "AIR.PA", "JPM", "TTE.PA"],
        "description": {
            "fr": "Mix ETF + blue chips solides (croissance modérée, dividendes)",
            "en": "ETF mix + solid blue chips (moderate growth, dividends)",
            "de": "ETF-Mix + solide Blue Chips (moderates Wachstum, Dividenden)",
            "es": "Mix ETF + blue chips sólidos (crecimiento moderado, dividendos)",
            "it": "Mix ETF + blue chip solide (crescita moderata, dividendi)",
        },
        "allocation": {"etf": 0.40, "blue_chip": 0.40, "growth": 0.20},
    },
    "agressif": {
        "tickers": ["NVDA", "TSLA", "META", "AMZN", "ASML.AS", "GOOGL", "MSFT", "MC.PA"],
        "description": {
            "fr": "Actions de croissance à fort potentiel (volatilité élevée, horizon long terme)",
            "en": "High-growth stocks (high volatility, long-term horizon)",
            "de": "Wachstumsaktien mit hohem Potenzial (hohe Volatilität, langer Horizont)",
            "es": "Acciones de alto crecimiento (alta volatilidad, horizonte largo plazo)",
            "it": "Azioni ad alta crescita (alta volatilità, orizzonte lungo termine)",
        },
        "allocation": {"growth": 0.70, "blue_chip": 0.30},
    },
}

ETF_TICKERS = {"CW8.PA", "ESE.PA", "IWDA.AS"}

# ── Seuils budget (€) ────────────────────────────────────────────────
BUDGET_THRESHOLDS = {
    "micro": 200,    # < 200€ → ETF fractionnés obligatoires
    "small": 500,    # 200–500€ → ETF + 1-2 actions abordables
    "medium": 2000,  # 500–2000€ → portefeuille diversifié
}


def get_market_data(ticker: str) -> dict | None:
    from data.fetcher import get_market_data as _gmd
    return _gmd(ticker)


# ══════════════════════════════════════════════════════════════════════
# ARCHITECTURE HUGGINGFACE — 3 TYPES D'API DISTINCTS
# ══════════════════════════════════════════════════════════════════════
#
# TYPE 1 — TEXT GENERATION via Chat Completions (format OpenAI-compatible)
#   Endpoint : /hf-inference/models/{model}/v1/chat/completions
#   Payload  : {"messages": [{"role": "user", "content": "..."}], "max_tokens": N}
#   Modèles  : Qwen2.5, Llama-3, Phi-3, Mistral — tout modèle d'instruction
#   Retour   : choices[0].message.content
#
# TYPE 2 — ZERO-SHOT CLASSIFICATION (pas besoin de fine-tuning)
#   Endpoint : /hf-inference/models/facebook/bart-large-mnli
#   Payload  : {"inputs": "texte", "parameters": {"candidate_labels": ["A","B"]}}
#   Modèles  : bart-large-mnli, DeBERTa-v3-large-mnli
#   Retour   : {labels: [...], scores: [...]} — trié par score décroissant
#
# TYPE 3 — TEXT CLASSIFICATION (fine-tuné sur labels fixes)
#   Endpoint : /hf-inference/models/ProsusAI/finbert
#   Payload  : {"inputs": "texte en anglais"}
#   Modèles  : finbert, roberta-sentiment
#   Retour   : [[{label, score}, ...]] — labels fixes du modèle
#
# ══════════════════════════════════════════════════════════════════════


# ── TYPE 1 : LLM via Chat Completions (OpenAI-compatible) ────────────
def _llm_chat(messages: list[dict], max_tokens: int = 500, temperature: float = 0.3) -> str | None:
    """
    Appel LLM HuggingFace via l'API chat/completions (format OpenAI-compatible).
    Tous les modèles d'instruction modernes supportent ce format sur HF router.

    Cascade de modèles — du plus puissant au plus léger :
    1. Qwen2.5-72B  : raisonnement financier excellent, multilingual, HF Pro
    2. Qwen2.5-7B   : bon équilibre qualité/latence, fonctionne sur free tier
    3. Phi-3.5-mini : 3.8B compact, raisonnement structuré, très rapide
    4. Llama-3.2-3B : plus petit, dernier recours avant fallback rule-based

    Format payload unifié (même pour tous les modèles ci-dessus) :
    {
        "messages": [
            {"role": "system", "content": "Instructions système..."},
            {"role": "user",   "content": "Question de l'utilisateur..."}
        ],
        "max_tokens": 500,
        "temperature": 0.3,
        "stream": False
    }
    """
    models = [
        "Qwen/Qwen2.5-72B-Instruct",        # Meilleur raisonnement, HF Pro
        "Qwen/Qwen2.5-7B-Instruct",          # Bon free tier, multilingual
        "microsoft/Phi-3.5-mini-instruct",    # 3.8B, raisonnement compact
        "meta-llama/Llama-3.2-3B-Instruct",  # Léger, dernier recours LLM
    ]

    for model in models:
        try:
            url = f"https://router.huggingface.co/hf-inference/models/{model}/v1/chat/completions"
            r = requests.post(
                url,
                headers={**HF_HEADERS, "Content-Type": "application/json"},
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=50,
            )
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if content and len(content) > 80:
                    return content
            # 503 = modèle en cours de chargement → passer au suivant
            elif r.status_code in (503, 429):
                continue
        except Exception:
            continue
    return None


# ── TYPE 2 : Zéro-shot classification (BART) ─────────────────────────
def _classify_zero_shot(text: str, candidate_labels: list[str]) -> str | None:
    """
    Classification zéro-shot via facebook/bart-large-mnli.
    Aucun fine-tuning requis — le modèle comprend les labels en langage naturel.

    Format payload DIFFÉRENT des LLMs :
    {
        "inputs": "Je veux investir 500€ en bourse",
        "parameters": {
            "candidate_labels": ["investir un budget", "analyser une action", ...],
            "multi_label": False  # une seule classe gagnante
        }
    }
    Retour : {"labels": ["investir un budget", ...], "scores": [0.87, ...]}
    → labels[0] est le plus probable
    """
    try:
        r = requests.post(
            "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli",
            headers=HF_HEADERS,
            json={
                "inputs": text[:512],
                "parameters": {
                    "candidate_labels": candidate_labels,
                    "multi_label": False,
                }
            },
            timeout=20,
        )
        if r.status_code == 200:
            result = r.json()
            labels = result.get("labels", [])
            if labels:
                return labels[0]  # Label avec le score le plus élevé
    except Exception:
        pass
    return None


def _lang_instruction(lang: str) -> str:
    instructions = {
        "fr": "Réponds UNIQUEMENT en français. Sois direct, structuré, professionnel.",
        "en": "Reply ONLY in English. Be direct, structured, professional.",
        "de": "Antworte NUR auf Deutsch. Sei direkt, strukturiert, professionell.",
        "es": "Responde SOLO en español. Sé directo, estructurado, profesional.",
        "it": "Rispondi SOLO in italiano. Sii diretto, strutturato, professionale.",
    }
    return instructions.get(lang, "Reply in the same language as the question.")


# ── Critères quantitatifs par profil de risque ───────────────────────
PROFILE_CRITERIA = {
    "prudent": {
        "max_pe": 22,           # P/E max acceptable
        "min_dividend": 0.02,   # rendement dividende min (2%)
        "max_beta": 0.85,       # beta max (faible sensibilité au marché)
        "max_drawdown": 0.25,   # perte maximale tolérée (25%)
        "min_market_cap": 20e9, # grande capitalisation uniquement
        "sectors_ok": ["Consumer Defensive", "Healthcare", "Utilities", "Financial Services"],
        "description": {
            "fr": "Capital protégé, revenus stables. P/E<22, dividende>2%, beta<0.85",
            "en": "Capital protected, stable income. P/E<22, dividend>2%, beta<0.85",
            "de": "Kapitalschutz, stabile Erträge. KGV<22, Dividende>2%, Beta<0.85",
            "es": "Capital protegido, ingresos estables. P/E<22, dividendo>2%, beta<0.85",
            "it": "Capitale protetto, reddito stabile. P/E<22, dividendo>2%, beta<0.85",
        },
    },
    "modéré": {
        "max_pe": 30,
        "min_dividend": 0.005,  # 0.5% min
        "max_beta": 1.2,
        "max_drawdown": 0.40,
        "min_market_cap": 10e9,
        "sectors_ok": ["Technology", "Healthcare", "Financial Services", "Consumer Defensive", "Consumer Cyclical", "Industrials"],
        "description": {
            "fr": "Croissance régulière + protection. P/E<30, beta<1.2, mix ETF/actions solides",
            "en": "Steady growth + protection. P/E<30, beta<1.2, ETF/solid stocks mix",
            "de": "Stetiges Wachstum + Schutz. KGV<30, Beta<1.2, ETF/Aktien-Mix",
            "es": "Crecimiento regular + protección. P/E<30, beta<1.2, mix ETF/acciones",
            "it": "Crescita regolare + protezione. P/E<30, beta<1.2, mix ETF/azioni",
        },
    },
    "agressif": {
        "max_pe": 60,           # P/E élevé toléré pour la croissance
        "min_dividend": 0.0,    # dividende non obligatoire
        "max_beta": 2.5,
        "max_drawdown": 0.65,
        "min_market_cap": 1e9,  # mid-cap autorisé
        "sectors_ok": ["Technology", "Consumer Cyclical", "Communication Services", "Healthcare"],
        "description": {
            "fr": "Fort potentiel de croissance. P/E jusqu'à 60, volatilité élevée acceptée, tech/innovation",
            "en": "High growth potential. P/E up to 60, high volatility accepted, tech/innovation",
            "de": "Hohes Wachstumspotenzial. KGV bis 60, hohe Volatilität akzeptiert, Tech/Innovation",
            "es": "Alto potencial de crecimiento. P/E hasta 60, alta volatilidad aceptada, tech/innovación",
            "it": "Alto potenziale di crescita. P/E fino a 60, alta volatilità accettata, tech/innovazione",
        },
    },
}


# ── Modèle 1 : Détection d'intention ─────────────────────────────────
INTENT_PATTERNS = {
    "invest_budget": [
        r'\b(\d+)\s*[€$£]', r'[€$£]\s*(\d+)',
        r'investir', r'invest', r'anlegen', r'invertir', r'investire',
        r'placer', r'mettre', r'budget',
    ],
    "analyze_ticker": [
        r'\$[A-Z]{2,5}', r'analyse[r]?', r'analysi[sz]', r'analysier',
        r'comment se porte', r'how is', r'que pense', r'what about',
    ],
    "compare": [
        r'vs\.?', r'compar[ei]', r'versus', r'ou (?:bien )?', r'or ',
        r'mieux que', r'better than', r'besser als',
    ],
    "portfolio_review": [
        r'mon portefeuille', r'my portfolio', r'mein portfolio', r'mi cartera',
        r'mon invest', r'my invest', r'rebalance', r'rééquilibr',
    ],
    "general_info": [],  # fallback
}


def _detect_intent(question: str, lang: str) -> str:
    """
    Modèle 1 — Détection d'intention (3 couches).

    Stratégie :
    1. BART zero-shot (TYPE 2 HF) : classification en langage naturel, sans règles
    2. Rule-based (fallback rapide) : regex patterns, toujours disponible

    BART comprend l'intention même dans des formulations nouvelles ou complexes.
    Le rule-based garantit une réponse même si HF est indisponible.
    """
    # ── Tentative BART zero-shot (comprend le contexte mieux que les regex) ──
    # Labels en anglais = meilleure performance BART (entraîné sur MNLI anglais)
    bart_labels = [
        "invest a specific budget amount in stocks or ETFs",
        "analyze or get details about a specific stock ticker",
        "compare two or more investment options",
        "review my personal investment portfolio",
        "general question about markets or finance",
    ]
    bart_to_intent = {
        "invest a specific budget amount in stocks or ETFs": "invest_budget",
        "analyze or get details about a specific stock ticker": "analyze_ticker",
        "compare two or more investment options": "compare",
        "review my personal investment portfolio": "portfolio_review",
        "general question about markets or finance": "general_info",
    }
    bart_result = _classify_zero_shot(question, bart_labels)
    if bart_result and bart_result in bart_to_intent:
        return bart_to_intent[bart_result]

    # ── Fallback rule-based (regex, instantané) ──────────────────────────────
    q = question.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, q, re.IGNORECASE):
                return intent
    return "general_info"


def _filter_by_criteria(assets: list[dict], profile_key: str) -> list[dict]:
    """
    Filtre les actifs selon les critères quantitatifs du profil.
    Retourne une liste triée par score de pertinence.
    """
    criteria = PROFILE_CRITERIA.get(profile_key, PROFILE_CRITERIA["modéré"])
    scored = []

    for a in assets:
        score = 0
        pe = a.get("pe_ratio")
        div = a.get("dividend_yield", 0) or 0
        sector = a.get("sector", "")
        change_1m = a.get("change_1m_pct", 0) or 0

        # Critère P/E
        if pe and pe <= criteria["max_pe"]:
            score += 2
        elif pe is None:  # ETF ou donnée manquante : neutre
            score += 1

        # Critère dividende
        if div >= criteria["min_dividend"]:
            score += 2
        elif profile_key == "agressif":
            score += 1  # dividende non requis pour agressif

        # Critère secteur
        if any(s in sector for s in criteria["sectors_ok"]):
            score += 2

        # Performance 1 mois (momentum)
        if change_1m > 0:
            score += 1
        if change_1m > 5:
            score += 1

        # ETF = toujours bon pour prudent/modéré
        if a.get("ticker") in ETF_TICKERS and profile_key != "agressif":
            score += 3

        scored.append((score, a))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored]


# ── Modèle 2 : Analyse marché ────────────────────────────────────────
def _llm_analyze_market(
    assets: list[dict], ticker: str | None, profile_key: str, lang: str,
    monetary_stance: str = "neutral"
) -> str | None:
    """
    Modèle 2 — Analyse des données marché pour le profil.
    Utilise l'API chat/completions HF (format OpenAI-compatible).

    Reçoit en entrée :
    - Les actifs filtrés et scorés selon le profil (P/E, dividende, secteur, momentum)
    - La stance de politique monétaire F1 (hawkish/dovish/neutral)

    Rôle : synthétiser les signaux quantitatifs en analyse qualitative,
    identifier les 2-3 actifs les plus pertinents pour ce profil précis.
    """
    criteria = PROFILE_CRITERIA.get(profile_key, PROFILE_CRITERIA["modéré"])
    criteria_desc = criteria["description"].get(lang, criteria["description"]["fr"])

    lines = []
    for a in assets[:6]:
        pe_str = f"P/E={a['pe_ratio']:.1f}" if a.get("pe_ratio") else "P/E=N/A"
        div_str = f"div={a['dividend_yield']*100:.1f}%" if a.get("dividend_yield") else ""
        m1 = f"1m={a['change_1m_pct']:+.1f}%" if a.get("change_1m_pct") is not None else ""
        country = a.get("country", "")
        lines.append(f"  {a['name']} ({a['ticker']}): {a['price']} | {pe_str} {div_str} {m1} | {country}")

    market_ctx = "\n".join(lines)

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a financial market analyst specializing in portfolio management. "
                f"Current monetary policy stance: {monetary_stance} (hawkish=restrictive, dovish=accommodative). "
                f"Answer in this language: {lang}. Be direct and structured, max 3 sentences."
            )
        },
        {
            "role": "user",
            "content": (
                f"Target risk profile: {profile_key} — {criteria_desc}\n\n"
                f"Market data (pre-filtered and scored for this profile):\n{market_ctx}\n\n"
                f"Identify the 2 most suitable assets for this profile. "
                f"Justify based on: P/E ratio, dividend yield, 1-month momentum, sector, "
                f"and current monetary policy stance ({monetary_stance})."
            )
        }
    ]
    return _llm_chat(messages, max_tokens=220, temperature=0.25)


# ── Modèle 3 : Génération de réponse adaptée au profil ───────────────
def _llm_profile_response(
    question: str, assets: list[dict], budget: int | None, profile_key: str,
    lang: str, intent: str, macro_ctx: str, tech_block: str, portfolio_block: str,
    analysis_text: str, monetary_stance: str = "neutral",
    news_topic: str = "general", news_consensus: float = 0.0,
) -> str | None:
    """
    Modèle 3 — Génération de la réponse finale adaptée au profil.

    Reçoit l'ensemble du contexte enrichi :
    - Analyse du Modèle 2 (actifs les plus adaptés)
    - F1 : stance politique monétaire (hawkish/dovish/neutral)
    - F2 : type d'événement dominant dans les news (earnings/M&A/regulatory...)
    - F3 : niveau de consensus entre modèles de sentiment

    Rôle : synthèse finale + conseil personnalisé au profil, budget, et langue.
    """
    criteria = PROFILE_CRITERIA.get(profile_key, PROFILE_CRITERIA["modéré"])
    criteria_desc = criteria["description"].get(lang, criteria["description"]["fr"])

    top_assets = "\n".join(
        f"  • {a['name']} ({a['ticker']}): {a['price']} "
        f"{'| P/E=' + str(round(a['pe_ratio'],1)) if a.get('pe_ratio') else ''} "
        f"| {'+' if (a.get('change_1m_pct') or 0) >= 0 else ''}{a.get('change_1m_pct', 'N/A')}% (1m) "
        f"| {a.get('country', '')}"
        for a in assets[:4]
    )

    # Construire le contexte enrichi pour le LLM
    context_parts = []
    if macro_ctx:
        context_parts.append(f"Macro context: {macro_ctx[:200]}")
    if monetary_stance != "neutral":
        context_parts.append(f"Central bank stance: {monetary_stance} (important for risk calibration)")
    if news_topic and news_topic != "general":
        context_parts.append(f"Dominant news type: {news_topic} (consensus: {round(news_consensus*100)}%)")
    if portfolio_block:
        context_parts.append(f"User portfolio:\n{portfolio_block}")
    if tech_block:
        context_parts.append(f"Technical signals: {tech_block}")
    if analysis_text:
        context_parts.append(f"Preliminary market analysis: {analysis_text}")

    budget_instruction = ""
    if budget:
        budget_instruction = (
            f"\nSTRICT BUDGET: {budget}€ MAXIMUM. "
            f"NEVER suggest buying a full share if its price exceeds {budget}€. "
            f"For small budgets, recommend fractional shares (Trade Republic, Revolut) "
            f"with exact allocation amounts."
        )

    intent_descriptions = {
        "invest_budget": "User wants to invest a specific amount.",
        "analyze_ticker": "User wants a detailed analysis of a specific stock.",
        "compare": "User wants to compare multiple investment options.",
        "portfolio_review": "User wants feedback on their existing portfolio.",
        "general_info": "General market or finance question.",
    }

    messages = [
        {
            "role": "system",
            "content": (
                f"You are Folio, an expert financial advisor. "
                f"Always respond in this language: {lang}. "
                f"Be direct, structured, and specific. Use bullet points when listing assets. "
                f"Always mention country exposure and 1-month performance when discussing assets."
            )
        },
        {
            "role": "user",
            "content": (
                f"Client profile: {profile_key} — {criteria_desc}\n"
                f"Intent: {intent_descriptions.get(intent, '')}{budget_instruction}\n\n"
                f"Question: \"{question}\"\n\n"
                f"Recommended assets for this profile (pre-scored):\n{top_assets}\n\n"
                + "\n".join(context_parts)
                + "\n\nProvide a direct, personalized response of 4-6 lines. "
                f"If a budget is given, include exact allocation amounts per asset. "
                f"Justify choices using the financial data provided."
            )
        }
    ]
    return _llm_chat(messages, max_tokens=450, temperature=0.3)


def _smart_advisor(
    snapshot: list[dict],
    budget: int | None,
    risk_profile: str,
    lang: str,
    question: str,
    macro_ctx: str = "",
) -> str:
    """
    Conseiller rule-based intelligent.
    Respecte le budget ET le profil de risque + critères quantitatifs.
    Utilisé quand le LLM échoue ou comme base de la réponse.
    """
    # Normaliser le profil
    profile_key = risk_profile.lower().replace("é", "e").replace("moderate", "modéré")
    if profile_key not in RISK_UNIVERSE:
        profile_key = "modéré"
    profile_key_display = risk_profile

    config = RISK_UNIVERSE.get(profile_key, RISK_UNIVERSE["modéré"])
    preferred_tickers = config["tickers"]
    profile_desc = config["description"].get(lang, config["description"]["fr"])

    # 1. Filtrer par tickers du profil
    profile_assets = [d for d in snapshot if d["ticker"] in preferred_tickers]
    if not profile_assets:
        profile_assets = snapshot[:5]

    # 2. Appliquer les critères quantitatifs du profil (P/E, dividende, secteur, momentum)
    profile_assets = _filter_by_criteria(profile_assets, profile_key)

    # Logique budget
    budget_tier = "large"
    if budget:
        if budget < BUDGET_THRESHOLDS["micro"]:
            budget_tier = "micro"
        elif budget < BUDGET_THRESHOLDS["small"]:
            budget_tier = "small"
        elif budget < BUDGET_THRESHOLDS["medium"]:
            budget_tier = "medium"

    # Construire la recommandation selon le budget
    if budget_tier == "micro" and budget:
        return _micro_budget_advice(profile_assets, budget, lang, profile_desc, profile_key_display, macro_ctx)
    elif budget_tier == "small" and budget:
        return _small_budget_advice(profile_assets, budget, lang, profile_desc, profile_key_display, macro_ctx)
    elif budget and budget_tier == "medium":
        return _medium_budget_advice(profile_assets, budget, lang, profile_desc, profile_key_display, macro_ctx)
    else:
        return _general_advice(profile_assets, lang, profile_desc, profile_key_display, macro_ctx)


def _micro_budget_advice(assets, budget, lang, profile_desc, risk_name, macro_ctx) -> str:
    """Conseil pour budget < 200€ : actions fractionnées / ETF."""
    etfs = [a for a in assets if a["ticker"] in ETF_TICKERS]
    others = [a for a in assets if a["ticker"] not in ETF_TICKERS][:2]
    top = (etfs + others)[:3]

    if lang == "fr":
        lines = [
            f"💡 **Profil {risk_name} — Budget {budget}€**",
            "",
            f"Avec {budget}€, la meilleure stratégie est d'investir via des **actions fractionnées** "
            f"(Trade Republic, Revolut) qui permettent d'acheter n'importe quelle action à partir de 1€.",
            "",
            f"🎯 Recommandation ({profile_desc}) :",
        ]
        for i, a in enumerate(top, 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            alloc = round(budget * [0.60, 0.30, 0.10][i-1] if i <= 3 else budget * 0.10)
            lines.append(f"  {i}. **{a['name']} ({a['ticker']})** — {a['price']} {a.get('currency','EUR')} "
                        f"({sign}{a['change_pct']}%) → allouer ~{alloc}€")
        if macro_ctx:
            lines += ["", f"📊 Contexte macro : {macro_ctx[:150]}"]
    elif lang == "en":
        lines = [
            f"💡 **{risk_name} profile — Budget €{budget}**",
            "",
            f"With €{budget}, the best strategy is **fractional shares** "
            f"(Trade Republic, Revolut) allowing you to buy any stock from €1.",
            "",
            f"🎯 Recommendation ({profile_desc}):",
        ]
        for i, a in enumerate(top, 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            alloc = round(budget * [0.60, 0.30, 0.10][i-1] if i <= 3 else budget * 0.10)
            lines.append(f"  {i}. **{a['name']} ({a['ticker']})** — {a['price']} {a.get('currency','')} "
                        f"({sign}{a['change_pct']}%) → allocate ~€{alloc}")
    elif lang == "de":
        lines = [
            f"💡 **{risk_name}-Profil — Budget {budget}€**",
            "",
            f"Mit {budget}€ empfehlen sich **Bruchteile von Aktien** "
            f"(Trade Republic, Revolut) — schon ab 1€ investierbar.",
            "",
            f"🎯 Empfehlung ({profile_desc}):",
        ]
        for i, a in enumerate(top, 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            alloc = round(budget * [0.60, 0.30, 0.10][i-1] if i <= 3 else budget * 0.10)
            lines.append(f"  {i}. **{a['name']} ({a['ticker']})** — {a['price']} {a.get('currency','')} "
                        f"({sign}{a['change_pct']}%) → ~{alloc}€ investieren")
    elif lang == "es":
        lines = [
            f"💡 **Perfil {risk_name} — Presupuesto {budget}€**",
            "",
            f"Con {budget}€, la mejor estrategia son las **acciones fraccionadas** "
            f"(Trade Republic, Revolut) desde 1€.",
            "",
            f"🎯 Recomendación ({profile_desc}):",
        ]
        for i, a in enumerate(top, 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            alloc = round(budget * [0.60, 0.30, 0.10][i-1] if i <= 3 else budget * 0.10)
            lines.append(f"  {i}. **{a['name']} ({a['ticker']})** — {a['price']} {a.get('currency','')} "
                        f"({sign}{a['change_pct']}%) → asignar ~{alloc}€")
    elif lang == "it":
        lines = [
            f"💡 **Profilo {risk_name} — Budget {budget}€**",
            "",
            f"Con {budget}€, la strategia migliore sono le **azioni frazionate** "
            f"(Trade Republic, Revolut) da 1€.",
            "",
            f"🎯 Raccomandazione ({profile_desc}):",
        ]
        for i, a in enumerate(top, 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            alloc = round(budget * [0.60, 0.30, 0.10][i-1] if i <= 3 else budget * 0.10)
            lines.append(f"  {i}. **{a['name']} ({a['ticker']})** — {a['price']} {a.get('currency','')} "
                        f"({sign}{a['change_pct']}%) → allocare ~{alloc}€")
    else:
        lines = [f"Budget €{budget} — {risk_name} profile:"]
        for a in top:
            sign = "+" if a["change_pct"] >= 0 else ""
            lines.append(f"• {a['name']} ({a['ticker']}) {sign}{a['change_pct']}%")

    return "\n".join(lines)


def _small_budget_advice(assets, budget, lang, profile_desc, risk_name, macro_ctx) -> str:
    """Budget 200–500€ : ETF prioritaire + 1-2 actions."""
    etfs = [a for a in assets if a["ticker"] in ETF_TICKERS][:1]
    stocks = [a for a in assets if a["ticker"] not in ETF_TICKERS][:2]
    top = etfs + stocks

    allocs = _compute_allocations(budget, len(top))

    if lang == "fr":
        lines = [
            f"💡 **Profil {risk_name} — Budget {budget}€**",
            "",
            f"Répartition recommandée ({profile_desc}) :",
        ]
        for i, (a, alloc) in enumerate(zip(top, allocs), 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            is_etf = "📦 ETF" if a["ticker"] in ETF_TICKERS else "📈"
            lines.append(
                f"  {i}. {is_etf} **{a['name']} ({a['ticker']})** "
                f"— {a['price']} {a.get('currency','EUR')} ({sign}{a['change_pct']}%) "
                f"→ **{alloc}€** ({round(alloc/budget*100)}%)"
            )
        lines += ["", "⚠️ Via Trade Republic ou Revolut pour les actions fractionnées."]
    elif lang == "de":
        lines = [f"💡 **{risk_name}-Profil — Budget {budget}€**", "", f"Empfohlene Aufteilung ({profile_desc}):"]
        for i, (a, alloc) in enumerate(zip(top, allocs), 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            lines.append(f"  {i}. **{a['name']} ({a['ticker']})** — {a['price']} ({sign}{a['change_pct']}%) → **{alloc}€**")
    elif lang == "es":
        lines = [f"💡 **Perfil {risk_name} — Presupuesto {budget}€**", "", f"Distribución recomendada ({profile_desc}):"]
        for i, (a, alloc) in enumerate(zip(top, allocs), 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            lines.append(f"  {i}. **{a['name']} ({a['ticker']})** — {a['price']} ({sign}{a['change_pct']}%) → **{alloc}€**")
    elif lang == "it":
        lines = [f"💡 **Profilo {risk_name} — Budget {budget}€**", "", f"Distribuzione raccomandata ({profile_desc}):"]
        for i, (a, alloc) in enumerate(zip(top, allocs), 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            lines.append(f"  {i}. **{a['name']} ({a['ticker']})** — {a['price']} ({sign}{a['change_pct']}%) → **{alloc}€**")
    else:
        lines = [f"💡 **{risk_name} profile — €{budget} budget**", "", f"Recommended allocation ({profile_desc}):"]
        for i, (a, alloc) in enumerate(zip(top, allocs), 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            lines.append(f"  {i}. **{a['name']} ({a['ticker']})** — {a['price']} ({sign}{a['change_pct']}%) → **€{alloc}**")

    return "\n".join(lines)


def _medium_budget_advice(assets, budget, lang, profile_desc, risk_name, macro_ctx) -> str:
    """Budget 500–2000€ : portefeuille diversifié."""
    top = assets[:4]
    allocs = _compute_allocations(budget, len(top))

    if lang == "fr":
        lines = [
            f"💡 **Profil {risk_name} — Budget {budget}€**",
            "",
            f"Portefeuille recommandé ({profile_desc}) :",
        ]
        for i, (a, alloc) in enumerate(zip(top, allocs), 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            label = "📦 ETF" if a["ticker"] in ETF_TICKERS else "📈"
            pct = round(alloc / budget * 100)
            lines.append(
                f"  {i}. {label} **{a['name']} ({a['ticker']})** "
                f"— {a['price']} {a.get('currency','EUR')} ({sign}{a['change_pct']}%) "
                f"→ **{alloc}€** ({pct}%)"
            )
        if macro_ctx:
            lines += ["", f"📊 {macro_ctx[:200]}"]
    else:
        labels_map = {
            "en": ("portfolio", "Recommended portfolio"),
            "de": ("Portfolio", "Empfohlenes Portfolio"),
            "es": ("cartera", "Cartera recomendada"),
            "it": ("portafoglio", "Portafoglio consigliato"),
        }
        _, header = labels_map.get(lang, ("portfolio", "Recommended portfolio"))
        lines = [f"💡 **{risk_name} — €{budget}**", "", f"{header} ({profile_desc}):"]
        for i, (a, alloc) in enumerate(zip(top, allocs), 1):
            sign = "+" if a["change_pct"] >= 0 else ""
            lines.append(f"  {i}. **{a['name']} ({a['ticker']})** — {a['price']} ({sign}{a['change_pct']}%) → €{alloc}")

    return "\n".join(lines)


def _general_advice(assets, lang, profile_desc, risk_name, macro_ctx) -> str:
    """Conseil général sans budget précis."""
    top = assets[:3]

    if lang == "fr":
        lines = [f"💡 **Profil {risk_name}** — {profile_desc}", ""]
        lines.append("🎯 Sélection du moment :")
        for a in top:
            sign = "+" if a["change_pct"] >= 0 else ""
            rec = a.get("recommendation", "")
            rec_str = f" | Analyste: {rec.upper()}" if rec else ""
            lines.append(
                f"  • **{a['name']} ({a['ticker']})** "
                f"— {a['price']} {a.get('currency','EUR')} ({sign}{a['change_pct']}%){rec_str}"
            )
        if macro_ctx:
            lines += ["", f"📊 {macro_ctx[:200]}"]
    elif lang == "de":
        lines = [f"💡 **{risk_name}** — {profile_desc}", "", "🎯 Aktuelle Auswahl:"]
        for a in top:
            sign = "+" if a["change_pct"] >= 0 else ""
            lines.append(f"  • **{a['name']} ({a['ticker']})** — {a['price']} ({sign}{a['change_pct']}%)")
    elif lang == "es":
        lines = [f"💡 **{risk_name}** — {profile_desc}", "", "🎯 Selección actual:"]
        for a in top:
            sign = "+" if a["change_pct"] >= 0 else ""
            lines.append(f"  • **{a['name']} ({a['ticker']})** — {a['price']} ({sign}{a['change_pct']}%)")
    elif lang == "it":
        lines = [f"💡 **{risk_name}** — {profile_desc}", "", "🎯 Selezione attuale:"]
        for a in top:
            sign = "+" if a["change_pct"] >= 0 else ""
            lines.append(f"  • **{a['name']} ({a['ticker']})** — {a['price']} ({sign}{a['change_pct']}%)")
    else:
        lines = [f"💡 **{risk_name}** — {profile_desc}", "", "🎯 Current selection:"]
        for a in top:
            sign = "+" if a["change_pct"] >= 0 else ""
            lines.append(f"  • **{a['name']} ({a['ticker']})** — {a['price']} ({sign}{a['change_pct']}%)")

    return "\n".join(lines)


def _compute_allocations(budget: int, n: int) -> list[int]:
    """Calcule des allocations décroissantes qui somment au budget."""
    if n == 0:
        return []
    if n == 1:
        return [budget]
    weights = [1 / (i + 1) for i in range(n)]
    total_w = sum(weights)
    allocs = [round(budget * w / total_w) for w in weights]
    # Ajustement pour arrondir proprement
    diff = budget - sum(allocs)
    allocs[0] += diff
    return allocs


def advise(
    question: str,
    portfolio: list | None = None,
    risk_profile: str = "modéré",
) -> dict:
    """
    Conseiller Folio v6 — pipeline 3 modèles multilingue.

    Modèle 1 (Intent)    → Détecte l'intention : investir / analyser / comparer / portefeuille
    Modèle 2 (Analyste)  → Analyse les données marché pour le profil, filtre par critères
    Modèle 3 (Conseiller)→ Génère la réponse finale adaptée au profil + budget + langue

    Le rule-based est toujours calculé en fallback fiable.
    """
    portfolio = portfolio or []

    # ── Modèle 1 : Langue + Intention ────────────────────────────────
    lang = detect_language(question)

    # Extraction budget
    amount_match = re.search(
        r'(\d[\d\s]{0,8})\s*[€$£]|[€$£]\s*(\d[\d\s]{0,8})|(\d[\d\s]{0,8})\s*(?:euros?|dollar|pound|livre)',
        question, re.IGNORECASE
    )
    amount = None
    budget_int = None
    if amount_match:
        raw = amount_match.group(1) or amount_match.group(2) or amount_match.group(3)
        if raw:
            amount = raw.replace(" ", "").strip()
            try:
                budget_int = int(amount)
            except ValueError:
                pass

    intent = _detect_intent(question, lang)

    # Normaliser profil
    profile_key = risk_profile.lower().replace("é", "e")
    if profile_key not in RISK_UNIVERSE:
        profile_key = "modéré"

    # ── Données en parallèle ──────────────────────────────────────────
    snapshot = []
    macro_ctx = ""
    market_news_list = []

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_snap  = ex.submit(get_universe_snapshot, 20)
        f_macro = ex.submit(get_macro_summary, lang)
        f_news  = ex.submit(get_market_news, lang, 4)
        snapshot         = f_snap.result()
        macro_ctx        = f_macro.result()
        market_news_list = f_news.result()

    # ── F1 : FOMC-RoBERTa — stance politique monétaire ───────────────
    # Analyse le contexte macro pour déterminer si Fed/BCE est hawkish ou dovish.
    # Ce signal influence directement les recommandations :
    # hawkish → prudence sur les actions de croissance (taux élevés = DCF défavorable)
    # dovish  → favorable aux actifs risqués et aux actions de croissance
    monetary_result = analyze_monetary_policy(macro_ctx, lang)
    monetary_stance = monetary_result.get("stance", "neutral")
    monetary_display = monetary_result.get("label_display", "")

    # Ticker mentionné dans la question
    ticker_mentioned = extract_ticker(question)
    news_sentiment = None
    technical = {}

    if ticker_mentioned:
        name = UNIVERSE.get(ticker_mentioned, ticker_mentioned)
        technical = get_technicals(ticker_mentioned)
        news_sentiment = get_ticker_news_sentiment(ticker_mentioned, name, user_lang=lang)

    # Analyse portefeuille
    portfolio_analysis = None
    portfolio_block = ""
    if portfolio:
        try:
            portfolio_analysis = analyze_portfolio(portfolio)
            portfolio_block = portfolio_summary_for_prompt(portfolio, lang=lang)
        except Exception:
            portfolio_block = _simple_portfolio_text(portfolio, lang)

    # ── Modèle 2 : Filtrage + Analyse marché ─────────────────────────
    # Filtrer le snapshot selon critères quantitatifs du profil
    config = RISK_UNIVERSE.get(profile_key, RISK_UNIVERSE["modéré"])
    preferred_tickers = config["tickers"]
    profile_assets = [d for d in snapshot if d["ticker"] in preferred_tickers]
    if not profile_assets:
        profile_assets = snapshot[:6]

    # Appliquer critères quantitatifs (P/E, dividende, secteur, momentum)
    scored_assets = _filter_by_criteria(profile_assets, profile_key)

    # Construire bloc technique
    tech_block = ""
    if technical and "error" not in technical:
        tech_block = format_signal(technical, lang=lang)
        if news_sentiment:
            tech_block += f" | {news_sentiment.get('summary', '')[:100]}"

    # Tentative Modèle 2 : analyse LLM des marchés
    # Intègre la stance monétaire F1 dans le raisonnement
    market_analysis = None
    try:
        market_analysis = _llm_analyze_market(
            scored_assets, ticker_mentioned, profile_key, lang,
            monetary_stance=monetary_stance
        )
    except Exception:
        pass

    # ── Rule-based advisor (fallback fiable) ─────────────────────────
    rule_based = _smart_advisor(
        snapshot=snapshot,
        budget=budget_int,
        risk_profile=risk_profile,
        lang=lang,
        question=question,
        macro_ctx=macro_ctx,
    )

    # ── Modèle 3 : Réponse finale adaptée au profil ──────────────────
    # Reçoit TOUS les signaux enrichis : F1 (monetary), F2 (topic), F3 (consensus)
    news_topic = news_sentiment.get("main_topic", "general") if news_sentiment else "general"
    news_consensus = news_sentiment.get("consensus_rate", 0.0) if news_sentiment else 0.0

    llm_final = None
    try:
        llm_final = _llm_profile_response(
            question=question,
            assets=scored_assets,
            budget=budget_int,
            profile_key=profile_key,
            lang=lang,
            intent=intent,
            macro_ctx=macro_ctx,
            tech_block=tech_block,
            portfolio_block=portfolio_block,
            analysis_text=market_analysis or "",
            monetary_stance=monetary_stance,
            news_topic=news_topic,
            news_consensus=news_consensus,
        )
    except Exception:
        pass

    # Sélection finale : LLM si qualité suffisante, sinon rule-based
    if llm_final and len(llm_final) > 120:
        final_answer = llm_final
        source = "llm"
    else:
        final_answer = rule_based
        source = "rule_based"

    # Assets mentionnés dans la réponse
    mentioned = []
    for d in snapshot:
        if d["ticker"] in final_answer or d["name"].split()[0].lower() in final_answer.lower():
            mentioned.append(d)

    return {
        "answer": final_answer,
        "lang": lang,
        "intent": intent,
        "amount": amount,
        "snapshot": snapshot[:8],
        "mentioned_assets": mentioned[:4],
        "news_sentiment": news_sentiment,
        "macro_context": macro_ctx,
        # F1 : stance politique monétaire Fed/BCE
        "monetary_policy": {
            "stance": monetary_stance,
            "display": monetary_display,
        },
        "technical": technical if technical and "error" not in technical else None,
        "portfolio_analysis": portfolio_analysis,
        "profile_criteria": PROFILE_CRITERIA.get(profile_key, {}),
        "answer_source": source,
    }


def get_portfolio_analysis(holdings: list[dict]) -> dict:
    if not holdings:
        return {"error": "Portefeuille vide"}
    return analyze_portfolio(holdings)


def _simple_portfolio_text(portfolio: list[dict], lang: str) -> str:
    lines = [f"  - {p.get('name', p.get('ticker', '?'))}: {p.get('amount', 0)}€" for p in portfolio]
    labels = {"fr": "Portefeuille :", "en": "Portfolio:", "de": "Portfolio:", "es": "Cartera:", "it": "Portafoglio:"}
    return labels.get(lang, "Portfolio:") + "\n" + "\n".join(lines)
