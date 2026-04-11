# Sentiment & Stock: Analyzing GDELT News Sentiment's Correlation with Stock Price Movements

**CMPT 733 — Big Data Lab 2 | Simon Fraser University | April 2026**

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
- These findings align with the Efficient Market Hypothesis and were independently corroborated by a parallel ML pipeline where a naive baseline outperformed a sentiment-augmented RandomForest model.

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
pip install -r requirements.txt
```

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

---

## License

This project was developed for academic purposes as part of CMPT 733 course at Simon Fraser University.
