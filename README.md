# SaaS Customer Churn Early-Warning System

An end-to-end machine learning project that predicts which SaaS customers are at risk of churning **before** they cancel, and explains *why* each customer is at risk using SHAP.

## Project Goal

Most churn models stop at a prediction. This project goes further: the core deliverable is an explainability layer that translates model output into plain-English, per-customer risk narratives — the kind of insight a Customer Success or Growth team could actually act on. Feature engineering (recency/frequency/monetary usage patterns, trend/slope features, engagement decay) is the primary differentiator, not just model accuracy.

## Problem Framing

Using monthly customer usage, billing, and support data, the model predicts the probability that a customer will churn in an upcoming period. Churn is defined as an **explicit cancellation event**.

## Dataset

[Synthetic SaaS Churn Sample](https://huggingface.co/datasets/arti199919/synthetic-saas-churn-sample) — Hugging Face

- ~84,800 rows of monthly panel data (one row per customer per month)
- Fields include plan type, MRR, session counts, feature usage score, support tickets, payment failures, NPS score, product incidents, and active seats
- Includes both a same-month churn flag and a forward-looking churn flag, which requires a careful leakage audit during EDA (see Phase 1 below)

## Approach

| Phase | Description |
|---|---|
| 0. Setup & Dataset | Repo structure, environment, dataset sourcing and churn label definition |
| 1. EDA & Leakage Audit | Class balance, missingness, time-based train/test split to prevent future information leakage |
| 2. Feature Engineering | R/F/M-style usage features, trend/slope features, engagement decay, tenure/cohort handling |
| 3. Modeling | Logistic Regression baseline, XGBoost/LightGBM main model, class-weighting vs. SMOTE for imbalance |
| 4. Explainability | Global and local SHAP analysis, dependence plots, plain-English "why this customer is at risk" narratives |
| 5. Packaging *(optional)* | Streamlit dashboard for live churn scoring and explanation |
| 6. Write-up *(optional)* | Full methodology and business-recommendation summary |

## Status

🚧 In progress — currently in Phase 0/1 (data sourcing, EDA, and leakage audit).

## Tech Stack

Python · pandas · scikit-learn · XGBoost/LightGBM · SHAP · (optional) Streamlit
