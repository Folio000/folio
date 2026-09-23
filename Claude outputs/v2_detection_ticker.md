# Folio v2 — Détection de ticker + données Yahoo Finance

**Date :** Septembre 2026  
**Objectif :** Enrichir l'analyse sentiment avec le cours réel du titre mentionné.

---

## Architecture

```
Utilisateur → texte libre
    ↓
Détection ticker (regex $AAPL ou dictionnaire "apple" → AAPL)
    ↓
yfinance → prix, variation, market cap, P/E
    ↓
FinBERT → sentiment
    ↓
Réponse JSON enrichie
```

## Modèles utilisés

| Modèle | Rôle | Source |
|--------|------|--------|
| `ProsusAI/finbert` | Analyse de sentiment financier | HuggingFace Inference API |

## Sources de données

| Source | Données fournies |
|--------|-----------------|
| **yfinance** (Yahoo Finance) | Prix clôture, variation %, capitalisation boursière, P/E ratio, volume, fourchette 52 semaines, secteur |

**yfinance** est une bibliothèque Python open source qui récupère les données de Yahoo Finance. Elle supporte les tickers US (AAPL, MSFT…) et européens (MC.PA pour LVMH, AIR.PA pour Airbus…).

## Dictionnaire ticker

Extraction en deux passes :
1. Regex `\$([A-Z]{1,5})` pour les symboles explicites comme `$AAPL`
2. Dictionnaire nom → ticker : `"apple" → "AAPL"`, `"lvmh" → "MC.PA"`, etc.

## Endpoint

```
POST /api/report
Body: { "text": "Apple a battu ses prévisions" }
Response: {
  "label": "positive",
  "score": 0.97,
  "ticker": "AAPL",
  "market": {
    "price": 189.5,
    "change_pct": 1.2,
    "market_cap": 2940000000000,
    "pe_ratio": 28.4
  }
}
```

## Limites

- Analyse uniquement le premier ticker trouvé dans le texte
- Pas de traduction (modèle FinBERT en anglais uniquement)
- Interface basique, pas de historique

---

*Ce fichier fait partie du dossier `historique/` documentant l'évolution de Folio.*
