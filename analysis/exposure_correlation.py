"""
BD733 - Q3: Exposure Correlation Analysis
Research Question: Is there correlation between media exposure (article count) and stock prices?
 
Method: Pearson correlation + hypothesis testing (per company)
        + Time-lagged correlations (1, 3, 7, 14 day lags)
Data: combined_data_clean from GCS bucket og-gdelt-main-data-dev
"""
 
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
            # We shift FORWARD to get future return (not backward)
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
# VISUALIZATIONS
# ============================================================
 
def plot_correlation_bars(results_df):
    """
    Bar chart comparing r-values across companies for both metrics.
    Same layout as Q1 tone script for consistency.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    companies = results_df["company"]
    x = np.arange(len(companies))
    width = 0.5
 
    # Exposure vs Next-Day Close
    colors_close = ["green" if r > 0 else "red" for r in results_df["exposure_vs_close_r"]]
    axes[0].bar(x, results_df["exposure_vs_close_r"], width, color=colors_close, alpha=0.7, edgecolor="black")
    axes[0].set_ylabel("Pearson r", fontsize=11)
    axes[0].set_title("Exposure vs Next-Day Close", fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(companies, fontsize=10)
    axes[0].axhline(y=0, color="black", linewidth=0.5)
    axes[0].set_ylim(-0.5, 0.5)
    for i, row in results_df.iterrows():
        if row["exposure_vs_close_significant"] == "Yes":
            axes[0].text(i, row["exposure_vs_close_r"] + 0.02, "*", ha="center", fontsize=14, fontweight="bold")
 
    # Exposure vs Daily Return
    colors_ret = ["green" if r > 0 else "red" for r in results_df["exposure_vs_return_r"]]
    axes[1].bar(x, results_df["exposure_vs_return_r"], width, color=colors_ret, alpha=0.7, edgecolor="black")
    axes[1].set_ylabel("Pearson r", fontsize=11)
    axes[1].set_title("Exposure vs Daily Return %", fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(companies, fontsize=10)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_ylim(-0.5, 0.5)
    for i, row in results_df.iterrows():
        if row["exposure_vs_return_significant"] == "Yes":
            axes[1].text(i, row["exposure_vs_return_r"] + 0.02, "*", ha="center", fontsize=14, fontweight="bold")
 
    fig.suptitle("Q3: Exposure Correlation — Pearson r by Company (* = p < 0.05)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "q3_exposure_correlation_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
 
 
def plot_scatter_per_company(df):
    """
    Scatter plots: exposure count vs daily return per company.
    Shows the raw data behind the correlation numbers.
    """
    companies = sorted(df["company"].unique())
    fig, axes = plt.subplots(1, len(companies), figsize=(6 * len(companies), 5))
 
    if len(companies) == 1:
        axes = [axes]
 
    for ax, company in zip(axes, companies):
        company_df = df[df["company"] == company]
        r, p = pearsonr(company_df["daily_exposure_count"], company_df["daily_return_pct"])
 
        ax.scatter(
            company_df["daily_exposure_count"],
            company_df["daily_return_pct"],
            alpha=0.2, s=10, color="#1f77b4"
        )
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Daily Exposure Count", fontsize=11)
        ax.set_ylabel("Daily Return %", fontsize=11)
        ax.set_title(f"{company}\nr={r:.4f}, p={p:.4f}", fontsize=12)
 
    fig.suptitle("Q3: Exposure vs Daily Return — Per Company", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "q3_exposure_vs_return_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
 
 
def plot_lagged_heatmap(lagged_df):
    """
    Heatmap showing r-values across companies and lag periods.
    Rows = companies, columns = lag days, color = correlation strength.
    Quick visual to see if any lag stands out.
    """
    pivot = lagged_df.pivot(index="company", columns="lag_days", values="r_value")
 
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
 
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{d}-day" for d in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_xlabel("Lag Period", fontsize=12)
    ax.set_title("Q3: Time-Lagged Exposure → Future Returns (Pearson r)", fontsize=13)
 
    # Add r-values as text on each cell
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            # Find if this cell is significant
            match = lagged_df[
                (lagged_df["company"] == pivot.index[i]) &
                (lagged_df["lag_days"] == pivot.columns[j])
            ]
            sig = match["significant"].values[0] if len(match) > 0 else "No"
            label = f"{val:.3f}{'*' if sig == 'Yes' else ''}"
            ax.text(j, i, label, ha="center", va="center", fontsize=10, fontweight="bold")
 
    plt.colorbar(im, ax=ax, label="Pearson r")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "q3_lagged_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
 
 
def plot_time_series(df):
    """
    Time series: exposure count + stock price overlaid per company (dual y-axes).
    Shows whether spikes in coverage coincide with price movements.
    """
    companies = sorted(df["company"].unique())
    fig, axes = plt.subplots(len(companies), 1, figsize=(14, 5 * len(companies)))
 
    if len(companies) == 1:
        axes = [axes]
 
    for ax, company in zip(axes, companies):
        company_df = df[df["company"] == company].sort_values("event_date")
        company_df = company_df.set_index("event_date")
        weekly = company_df.resample("W").agg({
            "daily_exposure_count": "mean",
            "Close": "mean"
        }).dropna()
 
        # Left y-axis: stock price
        color_price = "#1f77b4"
        ax.plot(weekly.index, weekly["Close"], color=color_price, linewidth=1.5, label="Close Price")
        ax.set_ylabel("Close Price ($)", color=color_price, fontsize=11)
        ax.tick_params(axis="y", labelcolor=color_price)
 
        # Right y-axis: exposure count
        ax2 = ax.twinx()
        color_exp = "#2ca02c"
        ax2.bar(weekly.index, weekly["daily_exposure_count"], width=5, color=color_exp, alpha=0.3, label="Avg Exposure")
        ax2.set_ylabel("Daily Exposure Count", color=color_exp, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=color_exp)
 
        ax.set_title(f"{company} — Weekly Avg Exposure vs Close Price (2020–2025)", fontsize=13)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
 
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
 
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "q3_exposure_vs_price_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
 
 
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
 
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_correlation_bars(results_df)
    plot_scatter_per_company(df)
    plot_lagged_heatmap(lagged_df)
    plot_time_series(df)
 
    print("\nDone! Check results/ folder for CSV and PNG outputs.")
 
 
if __name__ == "__main__":
    main()
 










