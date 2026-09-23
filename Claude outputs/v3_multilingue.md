# Folio v3 — Multilinguisme (FR/DE/ES/IT → EN)

**Date :** Septembre 2026  
**Objectif :** Permettre à l'utilisateur de saisir du texte en français, allemand, espagnol ou italien.

---

## Architecture

```
Utilisateur → texte (FR/DE/ES/IT/EN)
    ↓
Détection de langue (xlm-roberta)
    ↓ si non-anglais
Traduction vers l'anglais (Helsinki-NLP/opus-mt)
    ↓
Détection ticker + yfinance
    ↓
FinBERT (analyse sentiment en anglais)
    ↓
Réponse JSON
```

## Modèles utilisés

| Modèle | Rôle | Source |
|--------|------|--------|
| `papluca/xlm-roberta-base-language-detection` | Détection de langue | HuggingFace Inference API |
| `Helsinki-NLP/opus-mt-fr-en` | Traduction FR → EN | HuggingFace Inference API |
| `Helsinki-NLP/opus-mt-de-en` | Traduction DE → EN | HuggingFace Inference API |
| `Helsinki-NLP/opus-mt-es-en` | Traduction ES → EN | HuggingFace Inference API |
| `Helsinki-NLP/opus-mt-it-en` | Traduction IT → EN | HuggingFace Inference API |
| `ProsusAI/finbert` | Analyse de sentiment financier | HuggingFace Inference API |

### Détection de langue — xlm-roberta

XLM-RoBERTa est un modèle de représentation multilingue entraîné sur 100 langues. Le modèle fine-tuné de papluca retourne un score de probabilité pour chaque langue. On prend la langue avec le score le plus élevé.

### Traduction — Helsinki-NLP OPUS-MT

Série de modèles de traduction neuronale (NMT) open source de l'Université d'Helsinki. Chaque modèle est dédié à une paire de langues (fr→en, de→en, etc.). Ils sont légers et rapides via l'API HuggingFace.

## Sources de données

| Source | Données fournies |
|--------|-----------------|
| **yfinance** | Prix, variation %, fundamentaux |

## Limites

- Sentiment toujours calculé en anglais (après traduction)
- Pas de détection de langue dans la réponse affichée (réponse toujours en français)
- Pas de conseils d'investissement — seulement sentiment + données de marché

---

*Ce fichier fait partie du dossier `historique/` documentant l'évolution de Folio.*
