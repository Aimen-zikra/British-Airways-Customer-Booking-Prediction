# British Airways — Predicting Customer Booking Behaviour

> **Forage × British Airways | Data Science Virtual Experience | Task 2**  
> Framework: 12-Stage Data Analysis Process

---

## Project Overview

Customers today research and browse flights long before they buy — and often never return. By the time a customer arrives at the airport, British Airways has already lost the opportunity to influence that purchase decision.

This project builds a **machine learning model to predict which customers are likely to complete a booking**, enabling British AIrways to engage them proactively — before the booking window closes.

---

## Results at a Glance

| Metric | Score | Meaning |
|--------|-------|---------|
| **ROC-AUC** | **0.778** | Model correctly ranks a buyer above a non-buyer 77.8% of the time |
| **Recall** | **67.7%** | Correctly identifies 2 in 3 genuine buyers |
| **Accuracy** | **73.3%** | Overall correct classifications |
| **F1 Score** | **43.1%** | Balanced precision-recall score (reflects class imbalance) |
| **Precision** | **31.6%** | Expected given 85/15 class split |

> All metrics from **5-fold stratified cross-validation** — consistent across all folds (std < 0.004).

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Forage / British Airways |
| Rows | 50,000 customer booking records |
| Features | 14 original + 8 engineered |
| Target | `booking_complete` — 1 = booked, 0 = did not book |
| Class balance | 85% did not book / 15% booked |
| Missing values | None |

### Key Features

| Feature | Description |
|---------|-------------|
| `purchase_lead` | Days between booking date and travel date |
| `length_of_stay` | Nights at destination |
| `sales_channel` | Internet (88.8%) vs Mobile (11.2%) |
| `trip_type` | RoundTrip / OneWay / CircleTrip |
| `booking_origin` | Country of booking (104 unique) |
| `wants_extra_baggage` | Add-on selection flag |
| `flight_duration` | Total flight hours |

---

## Methodology

### Model: Random Forest Classifier

Chosen because it:
- Handles mixed numeric and categorical features natively
- Is robust to class imbalance via `class_weight='balanced'`
- Outputs built-in feature importances (required by the task)
- Scales efficiently on 50,000 rows with `n_jobs=-1`

### Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_estimators` | 200 | Stable feature importances |
| `max_depth` | 12 | Prevents overfitting |
| `min_samples_leaf` | 10 | Adds robustness |
| `max_features` | `'sqrt'` | Reduces tree correlation |
| `class_weight` | `'balanced'` | Handles 85/15 imbalance |
| `random_state` | 42 | Reproducibility |

### Class Balancing

Two complementary strategies:
1. **Upsampling** — minority class resampled to match majority (85,044 balanced training records)
2. **`class_weight='balanced'`** — further penalises minority class misclassification

The model was trained on the balanced set but **evaluated on the original imbalanced data** for honest real-world performance estimates.

---

## Feature Engineering

8 new features created from domain knowledge:

| Feature | Type | Rationale |
|---------|------|-----------|
| `total_add_ons` | Numeric | Sum of 3 add-on flags — purchase intent score |
| `origin_booking_rate` | Target-encoded | Average completion rate per country (most powerful) |
| `route_frequency` | Frequency | Popular routes attract more committed travellers |
| `lead_x_stay` | Interaction | Purchase lead × length of stay — the committed long-trip planner |
| `early_booker` | Binary | 1 if purchase_lead > 90 days |
| `long_stay` | Binary | 1 if length_of_stay > 14 nights |
| `is_weekend_flight` | Binary | Leisure vs business signal |
| `flight_day_num` | Ordinal | Mon=1 through Sun=7 |

---

## Top Feature Importances

| Rank | Feature | Importance | Business Meaning |
|------|---------|------------|-----------------|
| 1 | Booking Rate by Origin | **29.5%** | Geography is the #1 predictor of intent |
| 2 | Booking Origin (encoded) | 16.2% | Raw country also carries independent signal |
| 3 | Route Frequency | 8.6% | Popular routes = more committed travellers |
| 4 | Lead × Stay Interaction | 6.9% | Early planners on long trips are highly committed |
| 5 | Length of Stay | 6.3% | Longer trips = more considered decisions |
| 6 | Flight Duration | 5.8% | Longer flights → lower completion (price sensitivity) |
| 7 | Purchase Lead Time | 5.5% | Planning ahead correlates with follow-through |

---

## Key Findings

- **Geography dominates** — booking origin accounts for ~46% of model signal (origin + origin rate combined). Malaysia converts at 34.4%; Australia at 20.6%.
- **Add-ons signal intent** — customers who select extras are already in a buying mindset
- **Mobile has a 4.6pp gap** — 10.8% vs 15.5% for internet, suggesting UX friction not just different intent
- **Round Trip converts at 3× the rate** of One Way (15.1% vs 5.2%)
- **Linear models would struggle** — the strongest raw correlation with the target is only 0.106

---

## Recommendations

| # | Recommendation | Data Backing |
|---|----------------|-------------|
| 01 | **Geo-targeted marketing** — prioritise Malaysia & Australia | Origin = #1 feature at 29.5% |
| 02 | **Early-bird incentive programme** — reward bookings > 90 days out | `purchase_lead` & `lead_x_stay` in top 7 |
| 03 | **Bundle add-ons at checkout** — surface baggage + meal + seat together | Add-on customers convert at ~3× base rate |
| 04 | **Route-based campaign targeting** — focus spend on high-frequency routes | `route_frequency` = #3 feature at 8.6% |
| 05 | **Mobile funnel optimisation** — close the 4.6pp conversion gap | Mobile = 10.8% vs Internet = 15.5% |

---

## Tech Stack

- **Python 3.x**
- `pandas` — data manipulation
- `scikit-learn` — Random Forest, cross-validation, metrics
- `matplotlib` / `seaborn` — visualisation
- `numpy` — numerical operations

---
## How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/ba-booking-prediction.git
cd ba-booking-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Open the notebook
jupyter notebook BA_Customer_Booking_Prediction.ipynb
```
## Project Framework

This project follows a structured **12-stage data analysis process:**

```
01 Define Business Problem     →  07 Deep Analysis & Feature Engineering
02 Define Success Metrics      →  08 Model Training & Validation
03 Understand Data Landscape   →  09 Results & Insight Generation
04 Data Extraction             →  10 Validation & Sanity Checks
05 Data Cleaning               →  11 Recommendations
06 Exploratory Data Analysis   →  12 Presentation Guide
```


## Author

**Aimen Zikra** — Data Analyst  
Python · SQL · NLP · Data Visualization  
