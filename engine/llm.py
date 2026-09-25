"""
engine/llm.py — Groq LLM client pour Folio v8
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


SYSTEM_PROMPT = """Tu es Folio, un moteur d'éducation financière.
Tu expliques les dynamiques de marché, tu analyses l'impact des actualités sur les actions et tu fournis du contenu pédagogique équilibré.

Règles absolues :
- Ne recommande jamais d'acheter, de vendre ou de conserver un titre spécifique.
- Présente toujours les arguments haussiers et baissiers.
- Ancre chaque analyse dans les actualités récentes et le contexte macroéconomique disponibles.
- Sois précis et concis, sans formules creuses.
- Réponds dans la même langue que l'utilisateur.
- N'utilise ni émojis ni listes à tirets. Utilise des paragraphes structurés.

Structure ta réponse avec ces quatre sections exactes, séparées par une ligne vide :

CONTEXTE DE MARCHÉ
[état actuel du marché ou du titre : prix, variation, tendance]

ANALYSE FONDAMENTALE
[valorisation, fondamentaux, position sectorielle, catalyseurs récents]

ARGUMENTS HAUSSIERS / ARGUMENTS BAISSIERS
[deux paragraphes, l'un pour chaque camp, avec les arguments concrets]

RISQUES À SURVEILLER
[deux ou trois risques spécifiques identifiés dans les données ou l'actualité]

Tu n'es pas un conseiller financier. Tu fournis une analyse de marché à des fins éducatives."""


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

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Erreur LLM : {e}"
