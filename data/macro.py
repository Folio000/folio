"""
data/macro.py — Données macroéconomiques
Sources : FRED (Federal Reserve) + ECB (Banque Centrale Européenne)
Indicateurs : taux directeurs, inflation, courbe des taux, PIB
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
FRED_KEY = os.getenv("FRED_KEY", "")

# ── FRED Series IDs ──────────────────────────────────────────────────
FRED_SERIES = {
    # Taux
    "fed_rate":        "FEDFUNDS",        # Taux directeur Fed (%)
    "us_10y":          "DGS10",           # Rendement US 10 ans (%)
    "us_2y":           "DGS2",            # Rendement US 2 ans (%)
    "us_3m":           "DTB3",            # Rendement US 3 mois (%)
    # Inflation
    "cpi_us":          "CPIAUCSL",        # CPI USA (indice)
    "pce_us":          "PCEPI",           # PCE USA (inflation cible Fed)
    # Économie réelle
    "gdp_us":          "GDP",             # PIB US trimestriel (milliards $)
    "unemployment_us": "UNRATE",          # Taux de chômage US (%)
    # Marchés
    "vix":             "VIXCLS",          # VIX (volatilité S&P 500)
    "sp500":           "SP500",           # S&P 500 (niveau)
}

# ── ECB Statistical Data Warehouse ───────────────────────────────────
ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
ECB_SERIES = {
    "ecb_rate":   "FM/B.U2.EUR.RT0.BB.1100.M.R.A",   # Taux de dépôt BCE
    "eu_10y_de":  "FM/B.DE.EUR.FR.BB.GVB.1Y.M.B",     # Bund 10 ans Allemagne
    "hicp_eu":    "ICP/M.U2.N.000000.4.ANR",           # Inflation zone euro (HICP)
}


def _fred_latest(series_id: str, limit: int = 1) -> list[dict]:
    """Requête FRED : dernières observations pour une série."""
    if not FRED_KEY:
        return []
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": FRED_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            },
            timeout=10,
        )
        data = r.json()
        return data.get("observations", [])
    except Exception:
        return []


def _fred_value(series_id: str) -> float | None:
    """Retourne la dernière valeur numérique d'une série FRED."""
    obs = _fred_latest(series_id, limit=5)
    for o in obs:
        try:
            v = float(o["value"])
            return round(v, 3)
        except (ValueError, KeyError):
            continue
    return None


def _ecb_latest(series_key: str) -> float | None:
    """Requête ECB SDW : dernière valeur pour une clé de série."""
    try:
        r = requests.get(
            f"{ECB_BASE}/{series_key}",
            params={"format": "jsondata", "lastNObservations": 3},
            timeout=10,
        )
        data = r.json()
        # Structure ECB JSON : dataSets[0].series["0:0:0:..."}.observations
        datasets = data.get("dataSets", [])
        if not datasets:
            return None
        series = datasets[0].get("series", {})
        for key, val in series.items():
            obs = val.get("observations", {})
            if obs:
                # Clé = index temporel, valeur = [nombre, ...]
                latest_idx = max(obs.keys(), key=int)
                raw = obs[latest_idx][0]
                return round(float(raw), 3)
    except Exception:
        pass
    return None


def get_macro_data() -> dict:
    """
    Retourne un snapshot macro complet.
    {
        "us": { fed_rate, us_10y, us_2y, spread_10_2, cpi, gdp, unemployment, vix, sp500 },
        "eu": { ecb_rate, bund_10y, inflation },
        "yield_curve": "normale|inversée|plate",
        "context": str  # interprétation macro en français
    }
    """
    # ─── FRED ───────────────────────────────────────────────────────
    fed_rate      = _fred_value(FRED_SERIES["fed_rate"])
    us_10y        = _fred_value(FRED_SERIES["us_10y"])
    us_2y         = _fred_value(FRED_SERIES["us_2y"])
    us_3m         = _fred_value(FRED_SERIES["us_3m"])
    cpi_us        = _fred_value(FRED_SERIES["cpi_us"])
    unemployment  = _fred_value(FRED_SERIES["unemployment_us"])
    vix           = _fred_value(FRED_SERIES["vix"])

    # ─── ECB ────────────────────────────────────────────────────────
    ecb_rate  = _ecb_latest(ECB_SERIES["ecb_rate"])
    bund_10y  = _ecb_latest(ECB_SERIES["eu_10y_de"])
    hicp_eu   = _ecb_latest(ECB_SERIES["hicp_eu"])

    # ─── Courbe des taux US ─────────────────────────────────────────
    spread_10_2 = None
    yield_curve = "inconnue"
    if us_10y is not None and us_2y is not None:
        spread_10_2 = round(us_10y - us_2y, 3)
        if spread_10_2 > 0.25:
            yield_curve = "normale"
        elif spread_10_2 < -0.05:
            yield_curve = "inversée"
        else:
            yield_curve = "plate"

    # ─── Interprétation contextuelle ────────────────────────────────
    context_parts = []
    if fed_rate is not None:
        if fed_rate >= 5.0:
            context_parts.append(f"La Fed maintient des taux élevés ({fed_rate}%), pression sur les actions de croissance.")
        elif fed_rate <= 1.5:
            context_parts.append(f"Les taux Fed sont bas ({fed_rate}%), environnement favorable aux actifs risqués.")
        else:
            context_parts.append(f"Taux Fed à {fed_rate}%, contexte intermédiaire.")

    if yield_curve == "inversée":
        context_parts.append("La courbe des taux US est inversée — signal historique de récession dans 12–18 mois.")
    elif yield_curve == "normale":
        context_parts.append("Courbe des taux US normale — expansion économique probable.")

    if vix is not None:
        if vix > 30:
            context_parts.append(f"VIX élevé ({vix}) — forte volatilité, prudence conseillée.")
        elif vix < 15:
            context_parts.append(f"VIX bas ({vix}) — marché calme, faible prime de risque.")

    if ecb_rate is not None:
        context_parts.append(f"Taux BCE à {ecb_rate}% — impact sur les actions européennes et l'euro.")

    context = " ".join(context_parts) if context_parts else "Données macro en cours de chargement."

    return {
        "us": {
            "fed_rate": fed_rate,
            "yield_10y": us_10y,
            "yield_2y": us_2y,
            "yield_3m": us_3m,
            "spread_10_2": spread_10_2,
            "cpi": cpi_us,
            "unemployment": unemployment,
            "vix": vix,
        },
        "eu": {
            "ecb_rate": ecb_rate,
            "bund_10y": bund_10y,
            "inflation_hicp": hicp_eu,
        },
        "yield_curve": yield_curve,
        "context": context,
    }


def get_macro_summary(lang: str = "fr") -> str:
    """
    Résumé macro court dans la langue demandée.
    N'affiche que les données disponibles (filtre les None).
    Utilisé comme contexte dans le prompt LLM.
    """
    macro = get_macro_data()
    us = macro["us"]
    eu = macro["eu"]
    yc = macro["yield_curve"]

    # ── Construire uniquement avec les données disponibles ────────────
    parts = []

    # Taux Fed
    fed = us.get("fed_rate")
    if fed is not None:
        parts.append({"fr": f"Fed={fed}%", "en": f"Fed={fed}%", "de": f"Fed={fed}%", "es": f"Fed={fed}%", "it": f"Fed={fed}%"}.get(lang, f"Fed={fed}%"))

    # US 10 ans
    y10 = us.get("yield_10y")
    if y10 is not None:
        parts.append({"fr": f"US 10 ans={y10}%", "en": f"US10Y={y10}%", "de": f"US 10J={y10}%", "es": f"US 10a={y10}%", "it": f"US 10a={y10}%"}.get(lang, f"US10Y={y10}%"))

    # Courbe des taux (seulement si on a les données)
    if yc != "inconnue":
        curve_labels = {
            "normale":  {"fr": "courbe normale", "en": "normal curve", "de": "normale Kurve", "es": "curva normal", "it": "curva normale"},
            "inversée": {"fr": "courbe inversée", "en": "inverted curve", "de": "invertierte Kurve", "es": "curva invertida", "it": "curva invertita"},
            "plate":    {"fr": "courbe plate", "en": "flat curve", "de": "flache Kurve", "es": "curva plana", "it": "curva piatta"},
        }
        label = curve_labels.get(yc, {}).get(lang, yc)
        parts.append(label)

    # VIX
    vix = us.get("vix")
    if vix is not None:
        parts.append(f"VIX={vix}")

    # BCE
    ecb = eu.get("ecb_rate")
    if ecb is not None:
        parts.append({"fr": f"BCE={ecb}%", "en": f"ECB={ecb}%", "de": f"EZB={ecb}%", "es": f"BCE={ecb}%", "it": f"BCE={ecb}%"}.get(lang, f"ECB={ecb}%"))

    # Bund 10 ans
    bund = eu.get("bund_10y")
    if bund is not None:
        parts.append({"fr": f"Bund 10 ans={bund}%", "en": f"Bund 10Y={bund}%", "de": f"Bund 10J={bund}%", "es": f"Bund 10a={bund}%", "it": f"Bund 10a={bund}%"}.get(lang, f"Bund={bund}%"))

    if not parts:
        no_data = {
            "fr": "Données macro temporairement indisponibles.",
            "en": "Macro data temporarily unavailable.",
            "de": "Makrodaten vorübergehend nicht verfügbar.",
            "es": "Datos macro temporalmente no disponibles.",
            "it": "Dati macro temporaneamente non disponibili.",
        }
        return no_data.get(lang, no_data["en"])

    prefix = {"fr": "Macro : ", "en": "Macro: ", "de": "Makro: ", "es": "Macro: ", "it": "Macro: "}.get(lang, "Macro: ")
    summary = prefix + ", ".join(parts) + "."

    # Ajouter le contexte interprétatif si disponible
    if macro.get("context"):
        summary += " " + macro["context"]

    return summary
