# Folio v1 — Analyse de sentiment basique

**Date :** Septembre 2026  
**Objectif :** Prototype initial — analyser une phrase financière et retourner un sentiment.

---

## Architecture

```
Utilisateur → texte libre → API FastAPI → HuggingFace → label (positive/negative/neutral)
```

## Modèles utilisés

| Modèle | Rôle | Source |
|--------|------|--------|
| `ProsusAI/finbert` | Analyse de sentiment financier | HuggingFace Inference API |

**FinBERT** est un modèle BERT fine-tuné sur des textes financiers (articles de presse, rapports d'entreprise). Il retourne 3 labels : `positive`, `negative`, `neutral` avec un score de confiance.

## Sources de données

- Aucune donnée de marché réelle — analyse uniquement du texte entré par l'utilisateur.

## Endpoint

```
POST /api/report
Body: { "text": "Apple a battu ses prévisions" }
Response: { "label": "positive", "score": 0.97 }
```

## Limites

- Pas de données de marché (prix, variation)
- Pas de traduction (anglais uniquement)
- Pas de détection de ticker
- Interface minimale

---

*Ce fichier fait partie du dossier `historique/` documentant l'évolution de Folio.*
