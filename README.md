# Sentiment & Stock: Analyzing GDELT News Sentiment's Correlation with Stock Price Movements

**BD733 — Big Data Lab 2 | Simon Fraser University | April 2026**

**Team OG:** Dhruv Saikia, Kevin A. Gonzalez Casto, Ryan Peng, Sirathee Koomgreng

---

## Overview

This project investigates whether news sentiment from the [GDELT Project](https://www.gdeltproject.org/) (Global Database of Events, Language, and Tone) correlates with stock price movements. We analyze three companies across distinct sectors — **Amazon** (AMZN), **Pfizer** (PFE), and **Saudi Aramco** (2222.SR) — over the period **2020–2025**, using Pearson correlation and hypothesis testing.

### Research Questions

1. **Q1 — Tone Impact:** Does daily average news sentiment influence next-day stock price movement?
2. **Q2 — Theme Impact:** Which news themes affect stock prices most across different industries?
3. **Q3 — Exposure Correlation:** Is there a correlation between media exposure (daily article count) and stock prices?

### Key Findings

- News sentiment shows **no statistically significant correlation with daily stock returns** for any company (p > 0.05 for all).
- Apparent correlations with price *levels* are attributable to shared long-term trends (confounding variable), not predictive power.
- Only 5.9% of (company, theme) pairs showed significant correlation with daily returns — essentially noise.
- These findings align with the Efficient Market Hypothesis and were independently corroborated by a parallel ML pipeline (CMPT 756) where a naive baseline outperformed a sentiment-augmented RandomForest model.

---

## Repository Structure

```
gdelt-analysis/
├── README.md                          # This file
├── analysis/
│   ├── analysis.ipynb                 # Main Jupyter notebook — full analysis for Q1, Q2, Q3
│   ├── tone_prediction_analysis.py    # Q1: Tone correlation (standalone script)
│   ├── theme_importance.py            # Q2: Theme correlation (standalone script)
│   └── exposure_correlation.py        # Q3: Exposure correlation (standalone script)
├── scripts/
│   ├── extract_gdelt.py               # BigQuery GDELT extraction with entity resolution
│   ├── y_finance.py                   # Yahoo Finance stock data download
│   ├── join_data.py                   # BigQuery join: GDELT + stock prices
│   ├── clean.py                       # BigQuery cleaning + feature engineering
│   └── spark_train_gdelt.py           # PySpark ML training (for Dataproc comparison)
├── sql/
│   ├── tone_extract.sql               # SQL template: daily tone + exposure per company
│   └── themes_extract.sql             # SQL: daily theme mentions + tone per company per theme
├── src/
│   ├── __init__.py
│   ├── data_extractor.py              # DataExtractor class (extraction orchestration)
│   └── entity_resolver.py             # spaCy-based entity resolution for company names
├── config/
│   └── company_aliases.json           # Company name aliases (Amazon, Pfizer, Aramco)
├── files/                             # Raw data files (included for reproducibility)
│   ├── cleaned_data-combined_data_clean_000000000000-2.csv   # Tone + stock data (6,518 rows)
│   └── cleaned_data-themes_with_prices_clean_000000000000.csv # Themes + stock data (426,259 rows)
├── results/                           # Output directory for CSVs and charts (auto-generated)
├── extract_gkg_csv.py                 # GDELT GKG file crawler → S3 (Parquet conversion)
└── inspect_gkg_csv.py                 # GKG CSV zip extraction utility
```

---

## How to Run

### Prerequisites

- **Python 3.10+** (tested on 3.12)
- **pip** package manager

### 1. Clone the Repository

```bash
git clone https://github.com/rpeng35/gdelt-analysis.git
cd gdelt-analysis
```

### 2. Install Dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn jupyter yfinance
```

Full list of required packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥2.0 | Data loading, manipulation, aggregation |
| `numpy` | ≥1.24 | Numerical operations |
| `scipy` | ≥1.10 | `pearsonr` for Pearson correlation + p-values |
| `matplotlib` | ≥3.7 | Bar charts, scatter plots, time series |
| `seaborn` | ≥0.12 | Statistical visualization |
| `jupyter` | ≥1.0 | Notebook execution |
| `yfinance` | ≥0.2 | Stock price data (only needed for re-extraction) |

### 3. Run the Analysis Notebook

The primary deliverable is the Jupyter notebook. The data files are included in `files/`, so you can run the full analysis without any cloud credentials:

```bash
cd analysis
jupyter notebook analysis.ipynb
```

Run all cells sequentially. The notebook will:
1. Load the cleaned datasets from `./files/`
2. Run Pearson correlations for Q1 (tone), Q2 (themes), and Q3 (exposure)
3. Print statistical results with hypothesis test interpretations
4. Generate all visualizations (bar charts, scatter plots, time series)
5. Save result CSVs to `./results/`

### 4. Run Standalone Analysis Scripts (Alternative)

Each research question also has a standalone Python script that can be run independently:

```bash
# From the analysis/ directory:
python tone_prediction_analysis.py    # Q1: Tone impact
python theme_importance.py            # Q2: Theme impact
python exposure_correlation.py        # Q3: Exposure correlation
```

These scripts read from `data/` and write results + charts to `results/`.

> **Note:** The standalone scripts expect data in `data/` while the notebook expects data in `files/`. Adjust the paths in the script config sections if needed.

---

## Data Pipeline

The analysis uses pre-processed data that was produced by the following pipeline. You do **not** need to re-run the pipeline to test the analysis — the cleaned data is included in `files/`.

```
GDELT V2 GKG (petabytes)
    │
    ▼
BigQuery Extraction ──────────────── scripts/extract_gdelt.py
    │                                 + sql/tone_extract.sql
    │                                 + sql/themes_extract.sql
    │                                 + src/entity_resolver.py (spaCy NLP)
    ▼
Yahoo Finance ────────────────────── scripts/y_finance.py
    │                                 (AMZN, PFE, 2222.SR — 2020-2025)
    ▼
BigQuery Join ────────────────────── scripts/join_data.py
    │                                 (join on event_date + ticker)
    ▼
Cleaning + Feature Engineering ───── scripts/clean.py
    │                                 (forward-fill, LEAD/LAG, daily_return_pct)
    ▼
Two Clean CSVs in GCS
    │
    ├── combined_data_clean (6,518 rows × 13 cols, ~1 MB) ─── Q1 & Q3
    └── themes_with_prices_clean (426,259 rows × 15 cols, ~71 MB) ── Q2
```

### Dataset Columns

**combined_data_clean** (used for Q1 and Q3):

| Column | Description |
|--------|-------------|
| `event_date` | Date of the news articles |
| `company` | Company name (Amazon, Aramco, Pfizer) |
| `ticker` | Stock ticker (AMZN, 2222.SR, PFE) |
| `daily_exposure_count` | Number of GDELT articles mentioning the company that day |
| `daily_avg_tone` | Average V2Tone sentiment score across all articles that day |
| `Open, High, Low, Close, Volume` | Stock price data from yFinance |
| `next_day_close` | Next trading day's closing price (prediction target) |
| `daily_return_pct` | Percentage change in closing price from previous day |
| `day_of_week` | Day of the week (0=Monday, 6=Sunday) |

**themes_with_prices_clean** (used for Q2):

| Column | Description |
|--------|-------------|
| `event_date` | Date |
| `company`, `ticker` | Company identifiers |
| `theme_category` | GDELT V2Themes category (e.g., CYBER, AVIATION, MED) |
| `daily_theme_mentions` | Number of articles with this theme for this company that day |
| `daily_theme_avg_tone` | Average tone of articles with this theme |
| `Open, High, Low, Close, Volume` | Stock prices |
| `next_day_close` | Next day's closing price |

---

## Results Summary

### Q1 — Tone Impact

| Company | Tone vs Close (r) | p-value | Sig? | Tone vs Return (r) | p-value | Sig? |
|---------|-------------------|---------|------|---------------------|---------|------|
| Amazon | 0.1936 | < 0.0001 | Yes | -0.0223 | 0.2987 | No |
| Aramco | 0.2591 | < 0.0001 | Yes | 0.0339 | 0.1144 | No |
| Pfizer | 0.0778 | 0.0003 | Yes | 0.0126 | 0.5562 | No |

**Interpretation:** Tone correlates with price levels (shared long-term trends) but **not** with daily returns. The confounding variable is time — both tone and stock prices trend upward over 2020–2025.

### Q2 — Theme Impact

- **492** (company, theme) pairs tested with ≥30 observations each
- **270 (54.9%)** significant for price levels (next-day close)
- **Only 29 (5.9%)** significant for daily returns — barely above what chance would produce
- No single theme emerged as a reliable predictor across multiple companies

### Q3 — Exposure Correlation

| Company | Exposure vs Close (r) | p-value | Sig? | Exposure vs Return (r) | p-value | Sig? |
|---------|-----------------------|---------|------|------------------------|---------|------|
| Amazon | 0.0410 | 0.0560 | No | 0.0174 | 0.4188 | No |
| Aramco | -0.1712 | < 0.0001 | Yes | -0.0398 | 0.0636 | No |
| Pfizer | 0.1950 | < 0.0001 | Yes | 0.0335 | 0.1189 | No |

**Interpretation:** Same pattern as Q1 — exposure volume does not predict daily returns.

---

## Entity Resolution

A key technical challenge was mapping GDELT organization mentions to our target companies. The `src/entity_resolver.py` module uses **spaCy** (`en_core_web_sm`) with a `PhraseMatcher` to resolve name variations. Aliases are configured in `config/company_aliases.json`:

```json
{
  "Amazon": ["Amazon", "Amazon.com", "Amazon Inc", "AMZN", "AWS", "Amazon Web Services"],
  "Pfizer": ["Pfizer", "Pfizer Inc", "Pfizer Inc.", "PFE"],
  "Aramco": ["Saudi Aramco", "Aramco", "Saudi Arabian Oil Company", "Saudi Arabian Oil Co"]
}
```

These aliases are converted to regex patterns for BigQuery `REGEXP_CONTAINS` queries during extraction.

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Google BigQuery** | Querying petabyte-scale GDELT data, joining datasets, feature engineering |
| **Google Cloud Storage** | Raw and processed data storage |
| **Python (pandas, scipy)** | Statistical analysis (Pearson correlation, hypothesis testing) |
| **matplotlib, seaborn** | Visualization (bar charts, scatter plots, time series, heatmaps) |
| **spaCy** | NLP-based entity resolution for company name matching |
| **yfinance** | Stock price data extraction |
| **Jupyter Notebook** | Interactive analysis and reproducible results |
| **Streamlit** | Interactive dashboard (data product) |

---

## Known Limitations

1. **Entity ambiguity** — "Amazon" matches both the company and the Amazon rainforest/river in GDELT articles, introducing noise.
2. **Only 3 companies** — Results may not generalize across sectors or market caps.
3. **COVID-era volatility** — The 2020–2025 window includes extreme market events that can inflate spurious correlations.
4. **Daily aggregation granularity** — Sentiment may impact prices within hours; daily averaging washes out intraday signals.
5. **Non-company-specific tone** — GDELT V2Tone reflects article-level sentiment, not company-specific sentiment. An article mentioning multiple entities dilutes the signal.

---

## Future Work

- **Bonferroni correction** for multiple hypothesis testing across 492 theme/company pairs
- **Time-lagged correlation analysis** at 1, 3, 7, and 14 day horizons (partially implemented in `exposure_correlation.py`)
- **Expand dataset** to S&P 500 companies for broader generalizability
- **Social media sentiment** integration (Twitter/Reddit) for higher-frequency signals
- **Real-time streaming pipeline** with incremental BigQuery table updates

---

## GCP Project Details

For those with access to the shared GCP project:

| Resource | Value |
|----------|-------|
| Project ID | `gdelt-stock-sentiment-analysis` |
| GCS Bucket | `og-gdelt-main-data-dev` |
| Region | `us-west1` |
| BigQuery Dataset | `gdelt_analysis` |

---

## License

This project was developed for academic purposes as part of BD733 at Simon Fraser University.
