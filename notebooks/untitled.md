# Data Cleaning Plan for Mobile Prices Dataset
This document outlines the full cleaning strategy for the `Global_Mobile_Prices_2025_Extended.csv` dataset

## 1. Dataset Columns and Types

| Column              | Type      | Notes |
|--------------------|-----------|-------|
| brand              | object    | OK |
| model              | object    | OK |
| price_usd          | int64     | OK |
| ram_gb             | int64     | OK |
| storage_gb         | int64     | OK |
| camera_mp          | int64     | OK |
| battery_mah        | int64     | OK |
| display_size_inch  | float64   | OK |
| charging_watt      | int64     | OK |
| 5g_support         | object    | Convert to Boolean |
| os                 | object    | Normalize values |
| processor          | object    | Normalize + Extract CPU brand |
| rating             | float64   | OK |
| release_month      | object    | Convert to month number |
| year               | int64     | OK |

---

## 2. Missing Values
The dataset contains **no null values**.

We move directly to:
- Type corrections
- Cleaning inconsistencies
- Standardizing categorical fields
- Detecting outliers
- Checking duplicates

---

