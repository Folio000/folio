# Folio v4 — Dashboard UI + FinBERT amélioré

**Date :** Septembre 2026  
**Objectif :** Première vraie interface utilisateur avec dashboard de marché et historique.

---

## Architecture

```
Frontend (HTML/CSS/JS) ←→ FastAPI backend
    ↓
Marché : snapshot 12 tickers en parallèle (ThreadPoolExecutor)
    ↓
Analyse : phrase → langue → traduction → sentiment (FinBERT) → ticker → yfinance
    ↓
UI : dashboard cards + résultat sentiment + données ticker
```

## Modèles utilisés

| Modèle | Rôle | Source |
|--------|------|--------|
| `papluca/xlm-roberta-base-language-detection` | Détection de langue | HuggingFace Inference API |
| `Helsinki-NLP/opus-mt-{lang}-en` | Traduction multilingue | HuggingFace Inference API |
| `nickmuchi/finbert-tone-finetuned-finance-topic-detection` | Sentiment financier (principal) | HuggingFace Inference API |
| `ProsusAI/finbert` | Sentiment financier (fallback) | HuggingFace Inference API |

### Nouveau modèle : nickmuchi/finbert-tone

Fine-tuné sur des données financières avec les labels `bullish` / `bearish` / `neutral` (mappés vers `positive` / `negative` / `neutral`). Plus précis que FinBERT de base pour les textes de marché. Si ce modèle échoue, on tombe sur ProsusAI/finbert.

## Fetching parallèle avec ThreadPoolExecutor

Pour récupérer les données de 12 tickers simultanément sans attendre chaque requête yfinance séquentiellement :

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(get_market_data, t): t for t in tickers}
    for future in as_completed(futures):
        data = future.result()
        if data:
            results.append(data)
```

**Gain de temps :** ~6× plus rapide qu'un appel séquentiel (6 workers pour 12 tickers = 2 batchs).

## Sources de données

| Source | Données fournies |
|--------|-----------------|
| **yfinance** | Prix, variation %, cap boursière, P/E, volume, 52 semaines, secteur |
| **Yahoo Finance News** (via yfinance) | Titres d'actualité récents par ticker |

## Endpoints FastAPI

```
POST /api/report     → sentiment + données ticker
GET  /api/snapshot   → snapshot 12 tickers du marché
GET  /api/ticker/{symbol} → données d'un seul ticker
```

## UI

- Cards de marché en haut (snapshot)
- Zone de saisie + bouton Analyser
- Résultat : badge sentiment + jauge + données ticker
- Historique simple (liste de phrases analysées)
- Déployé sur **Railway** : `https://web-production-ebc6a.up.railway.app/`

## Limites identifiées (→ raison du pivot v5)

- L'outil analyse des *phrases* mais ne *conseille* pas
- Pas de réponse aux questions "J'ai 100€ à investir, que faire ?"
- Pas de profil de risque utilisateur
- Pas de portefeuille personnel
- Dashboard centré sur le sentiment, pas sur l'action

---

*Ce fichier fait partie du dossier `historique/` documentant l'évolution de Folio.*
