"""
BD733 - Q3: Exposure Correlation Analysis
Research Question: Is there correlation between media exposure (article count) and stock prices?
 
Method: Pearson correlation + hypothesis testing (per company)
        + Time-lagged correlations (1, 3, 7, 14 day lags)
Data: combined_data_clean from GCS bucket og-gdelt-main-data-dev
"""
 
import pandas as pd
from scipy.stats import pearsonr
import os
 
# ============================================================
# CONFIG - Update this path to wherever your data file lives
# ============================================================
DATA_PATH = os.path.join("data", "combined_data_clean_000000000000.csv")
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
 
def load_data(data_path):
    """Load the cleaned combined dataset."""
    df = pd.read_csv(data_path, parse_dates=["event_date"])
    df = df.dropna(subset=["daily_exposure_count", "next_day_close", "daily_return_pct"])
    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Companies: {df['company'].unique()}")
    print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    return df
 
 
def run_exposure_correlations(df):
    """
    Run Pearson correlation for each company:
      - daily_exposure_count vs next_day_close  (exposure vs price level)
      - daily_exposure_count vs daily_return_pct (exposure vs daily movement)
 
    Consistent with Q1 tone script — same two targets.
    """
    results = []
 
    for company in sorted(df["company"].unique()):
        company_df = df[df["company"] == company]
        n = len(company_df)
 
        # Exposure vs Next-Day Close (price level)
        r_close, p_close = pearsonr(
            company_df["daily_exposure_count"], company_df["next_day_close"]
        )
 
        # Exposure vs Daily Return % (daily movement)
        r_return, p_return = pearsonr(
            company_df["daily_exposure_count"], company_df["daily_return_pct"]
        )
 
        results.append({
            "company": company,
            "n_observations": n,
            "exposure_vs_close_r": round(r_close, 4),
            "exposure_vs_close_p": round(p_close, 6),
            "exposure_vs_close_significant": "Yes" if p_close < 0.05 else "No",
            "exposure_vs_return_r": round(r_return, 4),
            "exposure_vs_return_p": round(p_return, 6),
            "exposure_vs_return_significant": "Yes" if p_return < 0.05 else "No",
        })
 
    results_df = pd.DataFrame(results)
    return results_df
 
 
def run_lagged_correlations(df):
    """
    Time-lagged analysis: does today's exposure count correlate with
    stock returns 1, 3, 7, or 14 days later?
 
    This tests whether media exposure has a delayed effect on price.
    For example, a spike in news coverage today might move the stock
    not tomorrow but over the next week.
 
    Uses pearsonr for p-values, runs per-company.
    """
    lags = [1, 3, 7, 14]
    results = []
 
    for company in sorted(df["company"].unique()):
        company_df = df[df["company"] == company].sort_values("event_date").copy()
 
        for lag in lags:
            # Calculate the return over the next N days
            # pct_change(lag) looks BACKWARD, so we shift FORWARD to get future return
            company_df[f"future_return_{lag}d"] = (
                company_df["Close"].shift(-lag) - company_df["Close"]
            ) / company_df["Close"] * 100
 
            # Drop NaN rows created by the shift
            valid = company_df.dropna(subset=[f"future_return_{lag}d"])
 
            if len(valid) > 10:  # Need enough data points
                r, p = pearsonr(
                    valid["daily_exposure_count"], valid[f"future_return_{lag}d"]
                )
                results.append({
                    "company": company,
                    "lag_days": lag,
                    "n_observations": len(valid),
                    "r_value": round(r, 4),
                    "p_value": round(p, 6),
                    "significant": "Yes" if p < 0.05 else "No",
                })
 
    lagged_df = pd.DataFrame(results)
    return lagged_df
 
 
def print_results(results_df):
    """Print main correlation results with hypothesis test interpretation."""
    print("\n" + "=" * 70)
    print("Q3 RESULTS: Is there correlation between media exposure and stock prices?")
    print("=" * 70)
    print(f"Null hypothesis: No significant relationship between exposure and price")
    print(f"Significance level: alpha = 0.05")
    print("-" * 70)
 
    for _, row in results_df.iterrows():
        print(f"\n{row['company']} (n={row['n_observations']}):")
        print(f"  Exposure vs Next-Day Close:  r={row['exposure_vs_close_r']:.4f}, "
              f"p={row['exposure_vs_close_p']:.6f}  --> "
              f"{'REJECT null (significant)' if row['exposure_vs_close_significant'] == 'Yes' else 'FAIL TO REJECT null (not significant)'}")
        print(f"  Exposure vs Daily Return %:  r={row['exposure_vs_return_r']:.4f}, "
              f"p={row['exposure_vs_return_p']:.6f}  --> "
              f"{'REJECT null (significant)' if row['exposure_vs_return_significant'] == 'Yes' else 'FAIL TO REJECT null (not significant)'}")
 
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("  If significant for price levels but not returns = likely spurious")
    print("  (same shared-trend issue as Q1 tone analysis)")
    print("  If not significant for either = exposure does NOT correlate with price")
    print("=" * 70)
 
 
def print_lagged_results(lagged_df):
    """Print time-lagged correlation results."""
    print("\n" + "=" * 70)
    print("Q3 TIME-LAGGED ANALYSIS: Does exposure predict future returns?")
    print("=" * 70)
    print("Testing: does today's exposure count correlate with returns")
    print("         1, 3, 7, or 14 days into the future?")
    print("-" * 70)
 
    for company in sorted(lagged_df["company"].unique()):
        company_results = lagged_df[lagged_df["company"] == company]
        print(f"\n{company}:")
        for _, row in company_results.iterrows():
            sig_text = "SIGNIFICANT" if row["significant"] == "Yes" else "not significant"
            print(f"  {row['lag_days']:2d}-day lag:  r={row['r_value']:.4f}, "
                  f"p={row['p_value']:.6f}  --> {sig_text}")
 
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("  If no lag shows significance = exposure has no delayed effect on price")
    print("  If a specific lag is significant = exposure may have a delayed impact")
    print("=" * 70)
 
 
# ============================================================
# MAIN
# ============================================================
 
def main():
    print("=" * 70)
    print("BD733 — Q3: Exposure Correlation Analysis")
    print("=" * 70)
 
    # Load data
    df = load_data(DATA_PATH)
 
    # Run main correlations + hypothesis tests
    results_df = run_exposure_correlations(df)
    print_results(results_df)
 
    # Run time-lagged correlations
    lagged_df = run_lagged_correlations(df)
    print_lagged_results(lagged_df)
 
    # Save results to CSV (for use in Streamlit or report)
    results_path = os.path.join(OUTPUT_DIR, "q3_exposure_correlation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nMain results saved to: {results_path}")
 
    lagged_path = os.path.join(OUTPUT_DIR, "q3_exposure_lagged_results.csv")
    lagged_df.to_csv(lagged_path, index=False)
    print(f"Lagged results saved to: {lagged_path}")
 
    print("\nDone! Check results/ folder for CSV outputs.")
 
 
if __name__ == "__main__":
    main()