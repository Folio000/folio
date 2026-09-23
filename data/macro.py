"""
data/macro.py — Données macroéconomiques via FRED (Federal Reserve)
Clé API gratuite sur fred.stlouisfed.org
"""

import os
import requests

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fred(series_id: str) -> float | None:
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        return None
    try:
        r = requests.get(
            FRED_BASE,
            params={
                "series_id": series_id,
                "api_key": key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            },
            timeout=5,
        )
        if r.status_code == 200:
            obs = r.json().get("observations", [])
            if obs:
                val = obs[0].get("value", ".")
                return float(val) if val != "." else None
    except Exception:
        pass
    return None


def get_macro_summary() -> str:
    """
    Retourne un bloc textuel de données macro pour injection dans le prompt LLM.
    Fonctionne même sans clé FRED (renvoie une chaîne vide).
    """
    indicators = {
        "Taux Fed Funds (%)": "FEDFUNDS",
        "Inflation US — CPI (%)": "CPIAUCSL",
        "Chômage US (%)": "UNRATE",
        "PIB US — croissance (%)": "A191RL1Q225SBEA",
        "Taux 10 ans US (%)": "GS10",
    }

    lines = []
    for label, series_id in indicators.items():
        val = _fred(series_id)
        if val is not None:
            lines.append(f"{label} : {val:.2f}")

    if not lines:
        return ""

    return "\n".join(lines)
