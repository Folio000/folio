"""
engine/llm.py — Groq LLM client pour Welto v8
Modèle : openai/gpt-oss-20b
Rôle   : moteur d'éducation financière, jamais de conseil personnalisé
"""

import os
from groq import Groq

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("GROQ_API_KEY manquante")
        _client = Groq(api_key=key)
    return _client


SYSTEM_PROMPT = """Tu es Welto, un moteur d'éducation financière complète.
Tu couvres trois domaines : (1) les marchés financiers (actions, indices, obligations, matières premières, cryptos, ETF), (2) la fiscalité de l'épargne et des investissements (PEA, assurance-vie, plus-values, flat tax, TMI, prélèvements sociaux, IFI, succession, défiscalisation), et (3) l'éducation financière générale (budget, épargne, crédit, retraite, diversification).

Règles absolues :
- Ne recommande jamais d'acheter, de vendre ou de conserver un titre spécifique.
- Pour les analyses de marché, présente toujours les arguments haussiers et baissiers.
- Ancre chaque analyse dans les actualités récentes et le contexte macroéconomique disponibles.
- Sois précis et concis, sans formules creuses.
- Réponds dans la même langue que l'utilisateur.
- N'utilise ni émojis ni listes à tirets. Utilise des paragraphes structurés.
- Pour les questions fiscales, explique les mécanismes clairement et précise que les règles varient selon la situation personnelle — un conseiller fiscal ou notaire reste nécessaire pour des cas spécifiques.

Pour les analyses de marché, structure avec ces quatre sections séparées par une ligne vide :

CONTEXTE DE MARCHÉ
[état actuel du marché ou du titre : prix, variation, tendance]

ANALYSE FONDAMENTALE
[valorisation, fondamentaux, position sectorielle, catalyseurs récents]

ARGUMENTS HAUSSIERS / ARGUMENTS BAISSIERS
[deux paragraphes, un pour chaque camp, avec arguments concrets]

RISQUES À SURVEILLER
[deux ou trois risques spécifiques identifiés dans les données ou l'actualité]

Pour les questions fiscales et d'éducation financière générale, adapte librement la structure pour être le plus pédagogique possible — explique le mécanisme, les seuils, les cas pratiques, et les pièges courants.

Tu n'es pas un conseiller financier ni un avocat fiscaliste. Tu fournis une éducation financière objective."""


LANG_NAMES = {
    "fr": "français", "en": "English", "de": "Deutsch",
    "es": "español", "it": "italiano", "pt": "português",
    "nl": "Nederlands", "zh": "中文", "ja": "日本語", "ar": "العربية",
}


def ask(question: str, context: str, lang: str = "fr") -> str:
    """
    Envoie une question enrichie au LLM et retourne la réponse textuelle.
    context : bloc de données marché/news/macro injecté avant la question.
    lang    : code ISO de la langue détectée (ex: 'en', 'fr', 'de').
    """
    client = _get_client()
    lang_name = LANG_NAMES.get(lang, lang)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Données disponibles :\n{context}\n\n"
                f"Question : {question}\n\n"
                f"INSTRUCTION OBLIGATOIRE : ta réponse doit être rédigée UNIQUEMENT en {lang_name}. "
                f"Ne réponds dans aucune autre langue."
            ),
        },
    ]

    FALLBACK: dict[str, str] = {
        "fr": "Désolé, je n'ai pas pu générer une réponse. Veuillez reformuler votre question.",
        "en": "Sorry, I could not generate a response. Please try rephrasing your question.",
        "de": "Entschuldigung, ich konnte keine Antwort generieren. Bitte formulieren Sie Ihre Frage anders.",
        "es": "Lo siento, no pude generar una respuesta. Por favor reformule su pregunta.",
        "it": "Mi dispiace, non ho potuto generare una risposta. Riformuli la sua domanda.",
        "pt": "Desculpe, não consegui gerar uma resposta. Por favor, reformule sua pergunta.",
        "nl": "Sorry, ik kon geen antwoord genereren. Herformuleer uw vraag alstublieft.",
    }

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
        )
        content = completion.choices[0].message.content
        if not content or not content.strip():
            return FALLBACK.get(lang, FALLBACK["fr"])
        return content.strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return FALLBACK.get(lang, FALLBACK["fr"])
