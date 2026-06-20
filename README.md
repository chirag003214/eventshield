# 🛡️ EventShield — AI-Powered Event Traffic Management

**Gridlock Hackathon 2.0 | Flipkart × Bengaluru Traffic Police × HackerEarth**
**Theme:** Event-Driven Congestion (Planned & Unplanned)

---

## Problem

Political rallies, festivals, sports events, construction, and sudden incidents create localized
traffic breakdowns in Bengaluru. Today: event impact isn't quantified in advance, resource
deployment is experience-driven, and there's no post-event learning system.

## Solution

EventShield turns 8,173 historical Bengaluru traffic events into three operational capabilities,
each backed by a **cross-validated** model:

1. **Impact Prediction** — will an event be high-impact or prolonged (>2h)? *(AUC 0.87–0.88)*
2. **Hotspot Forecasting** — which police stations will see the most event load, by day & time window? *(R² 0.47, 36% better than baseline)*
3. **Resource Planning** — a rule-based engine that converts validated predictions into manpower, barricade, and diversion plans.

## What makes this honest (and defensible)

Most hackathon prototypes show impressive-looking but unvalidated numbers. EventShield does the opposite:

- **Every ML metric is 5-fold cross-validated** and shown in the in-app Model Card. Run `train_models.py` to reproduce them exactly.
- **We dropped a model that didn't work.** A direct resolution-time regressor scored R²≈0.03 — no better than guessing the median, because clear-times are heavy-tailed. We reframed it as a **binary prolonged-event classifier (>2h)** that actually works (AUC 0.88).
- **Resource numbers are labeled as heuristics**, not pretended to be ML outputs.
- **Limitations are stated openly** in the Model Card (dataset skew toward unplanned events, feature-signal caveats).

## Validated Model Performance

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| Prolonged-Event Classifier | Will event take >2h? | ROC-AUC | **0.88** |
| | | F1 | 0.74 |
| High-Impact Classifier | Needs serious response? | ROC-AUC | **0.87** |
| | | F1 | 0.78 |
| Hotspot Forecaster | Event count per station/window | R² | **0.47** |
| | | vs. baseline | **+36%** |

*All scores are out-of-fold (5-fold CV). Reproduce with `python train_models.py`.*

## App Pages

- **📊 Dashboard** — interactive map of all events + live feed
- **🔮 Impact Predictor** — enter event details, get validated prolonged/high-impact probabilities
- **📡 Hotspot Forecast** — predict event load per station for a chosen day & time window
- **👮 Resource Planner** — deployment recommendation driven by the predictions
- **📈 Analytics** — temporal, spatial, cause, and corridor breakdowns
- **🧪 Model Card** — full CV metrics, feature importance, and honest limitations

## Quick Start

```bash
pip install -r requirements.txt
python train_models.py     # reproduces models.pkl + metrics.json (optional; pre-built included)
streamlit run app.py
```

## Tech Stack

Python · Streamlit · Plotly (Mapbox) · scikit-learn (GradientBoosting) · pandas/numpy

## Dataset

ASTRAM / Bengaluru Traffic Police, via HackerEarth — 8,173 events, Nov 2023 – Apr 2024,
54 police stations, 22 corridors, 17 event causes.

## Key Data Insights

- Events peak at 4–6 AM and 7–10 PM (night operations window)
- Highest road-closure causes: VIP movement (80%), public events (46%), tree falls (39%)
- Top hotspot stations: Yelahanka, HAL Old Airport, Sadashivanagar
- 94% of records are unplanned incidents — planned-event prediction is lower-confidence

## Roadmap

- Real-time ATCS / camera feed integration for live congestion signals
- LLM-generated natural-language operational briefs
- Mobile field-officer app with GPS-tagged outcome logging (closes the learning loop)
- Public diversion alerts via Google Maps / Waze APIs
