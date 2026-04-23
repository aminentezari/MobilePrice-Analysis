# Mobile Phone Pricing & Performance Analysis

Data Visualization course project — Università degli Studi di Milano-Bicocca

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat&logo=powerbi&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## Overview

Analysis of global mobile phone prices across 379 models from 16 brands. The project examines how hardware specifications, branding, and market positioning influence pricing, using statistical modeling and interactive Power BI dashboards.

---

## Research Questions

- What factors most significantly influence mobile phone prices?
- How do hardware specs (RAM, Storage, Battery, Screen Size) impact pricing?
- Which brands command premium pricing, and how large is the brand effect?
- Does camera specification have a significant relationship with price?
- How do brand-specific interactions with hardware affect pricing strategies?

---

## Key Findings

**OLS Regression (R² = 0.816):**
| Feature | Price Impact |
|---|---|
| RAM | +$59.77 per GB |
| Storage | +$1.57 per GB |
| Battery Capacity | -$0.04 per mAh |
| Camera MP | -$0.16 per MP |

**Brand premium vs Apple (reference brand):**
- Sony commands +$353 above Apple pricing
- Vivo sits $541 below Apple pricing
- RAM is by far the strongest hardware predictor of price

**Outlier detection (Isolation Forest):** 19 outliers (5.01%) identified, mostly Apple and Samsung premium models

**Best value-for-money phones (≤ $400):** Xiaomi Poco M3 and Realme C25s topped the value score ranking

---

## Methodology

- **Dataset:** 379 mobile phones, 16 brands, 24 variables — sourced from Kaggle
- **Cleaning:** screen size string-to-numeric conversion, camera MP extraction, IQR outlier flagging
- **Derived variables:** `Price_per_RAM`, `Price_per_Storage`, `Camera_TotalMP`, `Value_Score`
- **Log transformation** applied to Price for OLS regression assumptions
- **Techniques:** Descriptive statistics, OLS regression, Correspondence Analysis, Isolation Forest

---

## Project Structure

```
data-viz-mobile-data/
│
├── README.md
├── notebooks/
│   └── mobile_price_analysis.ipynb   # Full analysis notebook
├── data/                              # Dataset (CSV)
└── powerbi/                          # Power BI dashboard files
```

---

## Tech Stack

- **Python:** pandas, numpy, statsmodels, scipy, matplotlib, seaborn, scikit-learn
- **Power BI:** interactive dashboards for brand and price category exploration
- **Dataset:** [Kaggle — Mobile Phone Specifications and Prices](https://www.kaggle.com)

---

*Data Visualization project — A.Y. 2024/2025 · Licensed under CC BY 4.0*
