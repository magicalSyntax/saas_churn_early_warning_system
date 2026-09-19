# SaaS Customer Churn Early-Warning System

An end-to-end machine learning project that predicts which SaaS customers are at risk of churning **before** they cancel, and explains *why* each customer is at risk using SHAP.

## Project Goal

Most churn models stop at a prediction. This project goes further: the core deliverable is an explainability layer that translates model output into plain-English, per-customer risk narratives, the kind of insight a Customer Success or Growth team could actually act on. Feature engineering (recency/frequency/monetary usage patterns, trend/slope features, engagement decay) is the primary differentiator, not just model accuracy.

## Problem Framing

Using monthly customer usage, billing, and support data, the model predicts the probability that a customer will churn in an upcoming period. Churn is defined as an **explicit cancellation event** (the `churned` column).

## Dataset

[Synthetic SaaS Churn Sample](https://huggingface.co/datasets/arti199919/synthetic-saas-churn-sample) on Hugging Face.

- 79,817 rows of monthly panel data (one row per customer per month), Jan 2024 to Dec 2025
- Fields include plan type (Starter, Pro, Business, Enterprise), MRR, session counts, feature usage score, support tickets, payment failures, NPS score, product incidents, and active seats
- Ships with both a same-month churn flag (`churned`) and a forward-looking flag (`churned_next_month`), which is a built-in leakage trap handled in Phase 1

> **Note:** this is synthetic data. Findings below describe *this* dataset and should not be read as general claims about SaaS churn.

## Approach

| Phase | Status | Description |
|---|---|---|
| 0. Setup & Dataset | Done | Repo structure, environment, dataset sourcing and churn label definition |
| 1. EDA & Leakage Audit | Done | Leakage audit, class balance, time-based split, pre-churn trend check |
| 2. Feature Engineering | Done | 34 frozen features: peer-relative level, trend, ratio, RFM-style, engagement decay |
| 3. Modeling | Next | Logistic Regression baseline, XGBoost/LightGBM, class-weighting vs. SMOTE |
| 4. Explainability | Planned | Global and local SHAP, dependence plots, plain-English risk narratives |
| 5. Packaging *(optional)* | Planned | Streamlit dashboard for live churn scoring and explanation |
| 6. Write-up *(optional)* | Planned | Full methodology and business-recommendation summary |

## Phase 1: EDA & Leakage Audit

### Leakage audit
`churned_next_month` is confirmed to be a pure one-row shift of `churned` within each `user_id` (match rate 1.0 once NaN and last-row artifacts are excluded; the dataset fills each user's final row with 0 instead of null, which explains an initial 93.7% raw match). The column was **dropped immediately** and is never used as a feature or a second target.

### Class balance
- Overall churn rate is **1.417%**, a severe imbalance. Accuracy is unusable, so **PR-AUC is the primary metric** (with ROC-AUC and precision/recall alongside).
- Monthly churn rate ranges roughly 0.8% to 2.6% across the 24-month panel with no strong secular trend. The Jan 2024 peak (2.56%) was checked and is not a dataset-start artifact.
- Churn varies by plan: Starter 1.76% vs Business 0.91%, roughly 2x.

### Time-based split
A random split would leak future information in panel data, so the split is by time:

| Split | Period | Rows | Churners |
|---|---|---|---|
| Train | Jan 2024 to Sep 2025 | 67,818 | 932 |
| Test | Oct 2025 to Dec 2025 | 11,999 | 199 |

With only 199 test churners, small metric differences between models are within noise, so Phase 3 reports bootstrap confidence intervals (resampling users, not rows).

### Pre-churn trend check
Comparing each churner-row to the same calendar month's population average (which removes seasonality) showed two effects:

1. **A large, persistent level gap.** Churners already sit about 9 sessions and 16 feature-usage-score points below the population average even 6 months before churning.
2. **A smaller widening of that gap** approaching churn (about 1 session and 3 usage points over the 6-month window).

So engagement *level relative to peers* is the dominant signal and decay is secondary. This reshaped Phase 2 priorities.

## Phase 2: Feature Engineering

### Design rule
Every feature for a row uses only that row and earlier rows for the same user. Nothing looks forward, and no full-history aggregate crosses the train/test boundary.

### Feature families

| Family | Examples | Idea |
|---|---|---|
| Peer-relative level | usage and sessions vs. that month's population | Strongest signal; also removes calendar effects |
| Trend / slope | 3- and 6-month slopes, 3-month vs. prior 3-month ratio | Captures engagement decay |
| Ratios | tickets per session, sessions per seat, MRR per seat | Normalizes raw counts across account sizes |
| RFM-style | months since payment failure, cumulative tickets, NPS carried forward | Recency, frequency, monetary framing |
| Engagement decay | exponentially weighted usage (short vs. long), usage vs. own peak | Single interpretable decay number for SHAP narratives |
| Tenure / plan | plan tier, new-user flag, zero-seat flag | Cohort effects |

### Data handling decisions
- **Zero-seat accounts:** 2,546 rows have zero active seats and churn at 5.66% (about 4x the base rate). They get an explicit `zero_seats` flag, and per-seat ratios are NaN there rather than dividing by ~0.
- **MRR:** `mrr` and `monthly_price` are identical columns. `mrr` is an account total (correlation 0.91 to 0.93 with active seats within each plan), so `mrr_per_seat` is a genuine price-per-seat feature.
- **Tenure:** `tenure_month` matches per-user row order exactly (range 0 to 23), so no user is left-censored and the signup cohort is a true signup cohort.
- **Missing values:** early-tenure rows have NaNs from rolling windows (up to 35.7% for 6-month features). Tree models handle these natively; Logistic Regression will use train-only median imputation with missing-indicator columns.

### Leakage controls
- **Truncation test:** for six per-user history features (cumulative tickets, incidents, payment failures, usage own-peak, short EWM, 6-month incident sum), recomputing on data truncated at a cutoff reproduces the full-dataset values exactly (6/6 match).
- **Two leaks caught before modeling:** `churned_shifted`, a one-row backward shift of the label left over from the leakage audit (P(next row churned) = 1.0), and duplicate `_vs_pop` columns from Phase 1. Both were removed.
- **Guardrails in code:** the feature-freeze step asserts that no label, ID, or suspiciously named column is in the feature set, and screens for any single feature above 0.90 AUC. The best legitimate feature is `usage_ewm_short` at 0.79.

### Feature selection
Univariate screening on train only found redundancy, so correlation clustering (absolute Spearman above 0.85) pruned the set. The usage-level signal turned out to be six near-identical features; it was reduced to two that answer different questions (`usage_ewm_short`, own recent level; `feature_usage_score_vs_peers`, gap to peers). MRR peer-comparison features mostly encoded plan tier and were dropped. The interaction `incident_x_usage_drop` showed no signal and was removed.

**Result: 34 features frozen** (`features_v1.json`).

### Findings so far (train set, 932 churners)

| Family | Best feature | PR-AUC lift vs. base rate | ROC-AUC |
|---|---|---|---|
| Usage level | `usage_ewm_short` | 3.3x | 0.79 |
| NPS | `nps_ffill` | 2.2x | 0.73 |
| Sessions level | `sessions_vs_peers` | 2.2x | 0.71 |
| Tickets | `tickets_3m_sum` | 2.0x | 0.66 |
| Pure slope features | `sessions_slope_3m` | 1.0x | 0.51 |

- **Level beats trend.** Peer-relative engagement level is the strongest signal, while slope features alone are near noise. This replicates the Phase 1 trend check by an independent method.
- **Some plausible SaaS signals are weak in this dataset:** seat contraction, payment failures, and product incidents (lift 1.0 to 1.3). This is a property of the synthetic data, not a general claim.
- **Caveat:** these are single-feature results on 932 churners, so small gaps (for example 2.18x vs. 2.08x) are not meaningful. Only the large gaps are. Whether trend features add value *in combination* with level features is tested in Phase 3.

## Status

In progress. Phases 0 to 2 are complete; Phase 3 (modeling) is next.

## Tech Stack

Python · pandas · scikit-learn · XGBoost/LightGBM · SHAP · (optional) Streamlit
