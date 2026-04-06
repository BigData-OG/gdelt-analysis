"""
BD733 - Q1: Tone Impact Analysis
Research Question: Does daily average news tone influence next-day stock price movement?
 
Method: Pearson correlation + hypothesis testing (per company)
Data: combined_data_clean from GCS bucket og-gdelt-main-data-dev
"""
 
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
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
    df = df.dropna(subset=["daily_avg_tone", "next_day_close", "daily_return_pct"])
    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Companies: {df['company'].unique()}")
    print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    return df
 
 
def run_tone_correlations(df):
    """
    Run Pearson correlation for each company:
      - daily_avg_tone vs next_day_close  (tone vs price level)
      - daily_avg_tone vs daily_return_pct (tone vs daily movement)
    
    pearsonr() returns (r_value, p_value) in one call.
    p < 0.05 = statistically significant = reject the null hypothesis.
    """
    results = []
 
    for company in sorted(df["company"].unique()):
        company_df = df[df["company"] == company]
        n = len(company_df)
 
        # Tone vs Next-Day Close (price level)
        r_close, p_close = pearsonr(
            company_df["daily_avg_tone"], company_df["next_day_close"]
        )
 
        # Tone vs Daily Return % (daily movement)
        r_return, p_return = pearsonr(
            company_df["daily_avg_tone"], company_df["daily_return_pct"]
        )
 
        results.append({
            "company": company,
            "n_observations": n,
            "tone_vs_close_r": round(r_close, 4),
            "tone_vs_close_p": round(p_close, 6),
            "tone_vs_close_significant": "Yes" if p_close < 0.05 else "No",
            "tone_vs_return_r": round(r_return, 4),
            "tone_vs_return_p": round(p_return, 6),
            "tone_vs_return_significant": "Yes" if p_return < 0.05 else "No",
        })
 
    results_df = pd.DataFrame(results)
    return results_df
 
 
def print_results(results_df):
    """Print results in a clear format with hypothesis test interpretation."""
    print("\n" + "=" * 70)
    print("Q1 RESULTS: Does daily news tone influence stock price movement?")
    print("=" * 70)
    print(f"Null hypothesis: No significant relationship between tone and price")
    print(f"Significance level: alpha = 0.05")
    print("-" * 70)
 
    for _, row in results_df.iterrows():
        print(f"\n{row['company']} (n={row['n_observations']}):")
        print(f"  Tone vs Next-Day Close:  r={row['tone_vs_close_r']:.4f}, "
              f"p={row['tone_vs_close_p']:.6f}  --> "
              f"{'REJECT null (significant)' if row['tone_vs_close_significant'] == 'Yes' else 'FAIL TO REJECT null (not significant)'}")
        print(f"  Tone vs Daily Return %:  r={row['tone_vs_return_r']:.4f}, "
              f"p={row['tone_vs_return_p']:.6f}  --> "
              f"{'REJECT null (significant)' if row['tone_vs_return_significant'] == 'Yes' else 'FAIL TO REJECT null (not significant)'}")
 
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("  Weak r with significant p for price levels = likely spurious")
    print("  (shared long-term trends, not tone predicting price)")
    print("  No significance for daily returns = tone does NOT predict movement")
    print("=" * 70)
 
 
# ============================================================
# VISUALIZATIONS
# ============================================================
 
def plot_time_series(df):
    """
    Time series chart: tone + stock price overlaid per company (dual y-axes).
    This is the key visual for showing whether tone and price move together.
    """
    companies = sorted(df["company"].unique())
    fig, axes = plt.subplots(len(companies), 1, figsize=(14, 5 * len(companies)))
 
    if len(companies) == 1:
        axes = [axes]
 
    for ax, company in zip(axes, companies):
        company_df = df[df["company"] == company].sort_values("event_date")
 
        # Resample to weekly averages so the chart isn't too noisy
        company_df = company_df.set_index("event_date")
        weekly = company_df.resample("W").agg({
            "daily_avg_tone": "mean",
            "Close": "mean"
        }).dropna()
 
        # Left y-axis: stock price
        color_price = "#1f77b4"
        ax.plot(weekly.index, weekly["Close"], color=color_price, linewidth=1.5, label="Close Price")
        ax.set_ylabel("Close Price ($)", color=color_price, fontsize=11)
        ax.tick_params(axis="y", labelcolor=color_price)
 
        # Right y-axis: tone
        ax2 = ax.twinx()
        color_tone = "#ff7f0e"
        ax2.plot(weekly.index, weekly["daily_avg_tone"], color=color_tone, linewidth=1, alpha=0.7, label="Avg Tone")
        ax2.set_ylabel("Daily Avg Tone", color=color_tone, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=color_tone)
 
        ax.set_title(f"{company} — Weekly Avg Tone vs Close Price (2020–2025)", fontsize=13)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
 
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
 
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "q1_tone_vs_price_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
 
 
def plot_correlation_bars(results_df):
    """
    Bar chart comparing r-values across companies for both metrics.
    Great for the presentation results slide.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    companies = results_df["company"]
    x = np.arange(len(companies))
    width = 0.5
 
    # Tone vs Close
    colors_close = ["green" if r > 0 else "red" for r in results_df["tone_vs_close_r"]]
    bars1 = axes[0].bar(x, results_df["tone_vs_close_r"], width, color=colors_close, alpha=0.7, edgecolor="black")
    axes[0].set_ylabel("Pearson r", fontsize=11)
    axes[0].set_title("Tone vs Next-Day Close", fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(companies, fontsize=10)
    axes[0].axhline(y=0, color="black", linewidth=0.5)
    axes[0].set_ylim(-0.5, 0.5)
    # Add significance stars
    for i, row in results_df.iterrows():
        if row["tone_vs_close_significant"] == "Yes":
            axes[0].text(i, row["tone_vs_close_r"] + 0.02, "*", ha="center", fontsize=14, fontweight="bold")
 
    # Tone vs Daily Return
    colors_ret = ["green" if r > 0 else "red" for r in results_df["tone_vs_return_r"]]
    bars2 = axes[1].bar(x, results_df["tone_vs_return_r"], width, color=colors_ret, alpha=0.7, edgecolor="black")
    axes[1].set_ylabel("Pearson r", fontsize=11)
    axes[1].set_title("Tone vs Daily Return %", fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(companies, fontsize=10)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_ylim(-0.5, 0.5)
    for i, row in results_df.iterrows():
        if row["tone_vs_return_significant"] == "Yes":
            axes[1].text(i, row["tone_vs_return_r"] + 0.02, "*", ha="center", fontsize=14, fontweight="bold")
 
    fig.suptitle("Q1: Tone Impact — Pearson Correlations by Company (* = p < 0.05)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "q1_tone_correlation_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
 
 
def plot_scatter_per_company(df):
    """
    Scatter plots: tone vs daily return per company.
    Shows the raw data behind the correlation numbers.
    """
    companies = sorted(df["company"].unique())
    fig, axes = plt.subplots(1, len(companies), figsize=(6 * len(companies), 5))
 
    if len(companies) == 1:
        axes = [axes]
 
    for ax, company in zip(axes, companies):
        company_df = df[df["company"] == company]
        r, p = pearsonr(company_df["daily_avg_tone"], company_df["daily_return_pct"])
 
        ax.scatter(
            company_df["daily_avg_tone"],
            company_df["daily_return_pct"],
            alpha=0.2, s=10, color="#1f77b4"
        )
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(x=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Daily Avg Tone", fontsize=11)
        ax.set_ylabel("Daily Return %", fontsize=11)
        ax.set_title(f"{company}\nr={r:.4f}, p={p:.4f}", fontsize=12)
 
    fig.suptitle("Q1: Tone vs Daily Return — Per Company", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "q1_tone_vs_return_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
 
 
# ============================================================
# MAIN
# ============================================================
 
def main():
    print("=" * 70)
    print("BD733 — Q1: Tone Impact Analysis")
    print("=" * 70)
 
    # Load data
    df = load_data(DATA_PATH)
 
    # Run correlations + hypothesis tests
    results_df = run_tone_correlations(df)
 
    # Print results
    print_results(results_df)
 
    # Save results to CSV (for use in Streamlit or report)
    results_path = os.path.join(OUTPUT_DIR, "q1_tone_correlation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
 
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_time_series(df)
    plot_correlation_bars(results_df)
    plot_scatter_per_company(df)
 
    print("\nDone! Check the results/ folder for outputs.")
 
 
if __name__ == "__main__":
    main()