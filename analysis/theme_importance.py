"""
BD733 - Q2: Theme Impact Analysis
Research Question: Which news themes affect stock prices most across different industries?
 
Method: Pearson correlation + hypothesis testing (per company, per theme)
Data: themes_with_prices_clean from GCS bucket og-gdelt-main-data-dev
 
NOTE: This script uses the THEMES dataset (71MB, ~492,789 rows)
      not the combined_data dataset used by Q1 and Q3.
"""
 
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
 
# ============================================================
# CONFIG - Update this path to wherever your data file lives
# ============================================================
DATA_PATH = os.path.join("data", "themes_with_prices_clean_000000000000.csv")
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
 
def load_data(data_path):
    """Load the cleaned themes dataset."""
    df = pd.read_csv(data_path, parse_dates=["event_date"])
    df = df.dropna(subset=["daily_theme_avg_tone", "next_day_close"])
 
    # Calculate daily_return_pct if it doesn't exist in themes data
    # (themes dataset may not have this column pre-calculated)
    if "daily_return_pct" not in df.columns:
        df = df.sort_values(["ticker", "event_date"])
        df["daily_return_pct"] = df.groupby("ticker")["Close"].pct_change() * 100
        df = df.dropna(subset=["daily_return_pct"])
 
    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Companies: {df['company'].unique()}")
    print(f"Theme categories: {df['theme_category'].nunique()} unique themes")
    print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    return df
 
 
def run_theme_correlations(df, min_observations=30):
    """
    For each company, correlate each theme's tone with stock metrics.
 
    For every (company, theme) pair:
      - daily_theme_avg_tone vs next_day_close
      - daily_theme_avg_tone vs daily_return_pct
 
    min_observations: skip theme/company combos with fewer than this
                      many data points (too few = unreliable correlation)
    """
    results = []
 
    for company in sorted(df["company"].unique()):
        company_df = df[df["company"] == company]
 
        for theme in company_df["theme_category"].unique():
            theme_df = company_df[company_df["theme_category"] == theme]
            n = len(theme_df)
 
            # Skip if too few data points for a reliable correlation
            if n < min_observations:
                continue
 
            # Theme tone vs Next-Day Close
            r_close, p_close = pearsonr(
                theme_df["daily_theme_avg_tone"], theme_df["next_day_close"]
            )
 
            # Theme tone vs Daily Return %
            r_return, p_return = pearsonr(
                theme_df["daily_theme_avg_tone"], theme_df["daily_return_pct"]
            )
 
            results.append({
                "company": company,
                "theme_category": theme,
                "n_observations": n,
                "theme_vs_close_r": round(r_close, 4),
                "theme_vs_close_p": round(p_close, 6),
                "theme_vs_close_significant": "Yes" if p_close < 0.05 else "No",
                "theme_vs_return_r": round(r_return, 4),
                "theme_vs_return_p": round(p_return, 6),
                "theme_vs_return_significant": "Yes" if p_return < 0.05 else "No",
            })
 
    results_df = pd.DataFrame(results)
    return results_df
 
 
def get_top_themes(results_df, n=10):
    """
    Extract top N themes by absolute correlation strength, per company.
    Uses daily_return_pct correlation (the more meaningful metric).
    """
    top_themes = {}
 
    for company in sorted(results_df["company"].unique()):
        company_results = results_df[results_df["company"] == company].copy()
        company_results["abs_r_return"] = company_results["theme_vs_return_r"].abs()
        top = company_results.nlargest(n, "abs_r_return")
        top_themes[company] = top
 
    return top_themes
 
 
def print_results(results_df, top_themes):
    """Print results with top themes per company."""
    print("\n" + "=" * 70)
    print("Q2 RESULTS: Which news themes affect stock prices most?")
    print("=" * 70)
    print(f"Null hypothesis: No significant relationship between theme tone and price")
    print(f"Significance level: alpha = 0.05")
 
    # Summary stats
    total_pairs = len(results_df)
    sig_close = len(results_df[results_df["theme_vs_close_significant"] == "Yes"])
    sig_return = len(results_df[results_df["theme_vs_return_significant"] == "Yes"])
    print(f"\nTotal (company, theme) pairs tested: {total_pairs}")
    print(f"Significant for next-day close: {sig_close} ({sig_close/total_pairs*100:.1f}%)")
    print(f"Significant for daily return:   {sig_return} ({sig_return/total_pairs*100:.1f}%)")
 
    print("\n" + "-" * 70)
 
    for company, top in top_themes.items():
        print(f"\n{company} — Top 10 themes by |correlation| with daily return:")
        print(f"  {'Theme':<35} {'r':>8} {'p':>10} {'Sig?':>5}")
        print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*5}")
        for _, row in top.iterrows():
            print(f"  {row['theme_category']:<35} {row['theme_vs_return_r']:>8.4f} "
                  f"{row['theme_vs_return_p']:>10.6f} {row['theme_vs_return_significant']:>5}")
 
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    print("  Compare top themes across companies to see sector differences")
    print("  Tech (Amazon) may respond to different themes than Pharma (Pfizer)")
    print("  Few significant themes for daily returns = themes don't predict movement")
    print("=" * 70)
 
 
def print_cross_company_comparison(results_df):
    """
    Show which themes appear as significant across multiple companies.
    If a theme is significant for 2+ companies, it may be a universal driver.
    """
    print("\n" + "=" * 70)
    print("Q2 CROSS-COMPANY: Themes significant for daily returns in 2+ companies")
    print("=" * 70)
 
    sig_only = results_df[results_df["theme_vs_return_significant"] == "Yes"]
    theme_counts = sig_only.groupby("theme_category")["company"].count()
    shared_themes = theme_counts[theme_counts >= 2]
 
    if len(shared_themes) == 0:
        print("\nNo themes are significant for daily returns across 2+ companies.")
        print("This suggests theme impact is company/sector-specific, not universal.")
    else:
        print(f"\n{len(shared_themes)} themes significant across multiple companies:")
        for theme, count in shared_themes.sort_values(ascending=False).items():
            companies = sig_only[sig_only["theme_category"] == theme]["company"].tolist()
            print(f"  {theme}: {', '.join(companies)}")
 
    print("=" * 70)
 
 
# ============================================================
# VISUALIZATIONS
# ============================================================
 
def plot_top_themes_per_company(top_themes):
    """
    Horizontal bar chart of top 10 themes per company by correlation strength.
    """
    companies = sorted(top_themes.keys())
    fig, axes = plt.subplots(1, len(companies), figsize=(8 * len(companies), 6))
 
    if len(companies) == 1:
        axes = [axes]
 
    for ax, company in zip(axes, companies):
        top = top_themes[company].sort_values("theme_vs_return_r")
        colors = ["green" if r > 0 else "red" for r in top["theme_vs_return_r"]]
 
        ax.barh(top["theme_category"], top["theme_vs_return_r"], color=colors, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Pearson r", fontsize=11)
        ax.set_title(f"{company}\nTop 10 Themes", fontsize=12)
        ax.axvline(x=0, color="black", linewidth=0.5)
 
        # Add stars for significant themes
        for i, (_, row) in enumerate(top.iterrows()):
            if row["theme_vs_return_significant"] == "Yes":
                x_pos = row["theme_vs_return_r"]
                offset = 0.005 if x_pos >= 0 else -0.005
                ax.text(x_pos + offset, i, "*", va="center", fontsize=12, fontweight="bold")
 
    fig.suptitle("Q2: Top Themes by Correlation with Daily Return (* = p < 0.05)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "q2_top_themes_per_company.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
 
 
def plot_significance_summary(results_df):
    """
    Bar chart showing how many themes are significant per company
    for each metric (next-day close vs daily return).
    """
    summary = []
    for company in sorted(results_df["company"].unique()):
        company_data = results_df[results_df["company"] == company]
        total = len(company_data)
        sig_close = len(company_data[company_data["theme_vs_close_significant"] == "Yes"])
        sig_return = len(company_data[company_data["theme_vs_return_significant"] == "Yes"])
        summary.append({
            "company": company,
            "total_themes": total,
            "sig_close": sig_close,
            "sig_return": sig_return,
            "pct_sig_close": round(sig_close / total * 100, 1) if total > 0 else 0,
            "pct_sig_return": round(sig_return / total * 100, 1) if total > 0 else 0,
        })
 
    summary_df = pd.DataFrame(summary)
 
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary_df))
    width = 0.35
 
    ax.bar(x - width / 2, summary_df["pct_sig_close"], width, label="vs Next-Day Close", color="#1f77b4", alpha=0.7)
    ax.bar(x + width / 2, summary_df["pct_sig_return"], width, label="vs Daily Return %", color="#ff7f0e", alpha=0.7)
 
    ax.set_ylabel("% of Themes Significant (p < 0.05)", fontsize=11)
    ax.set_title("Q2: How Many Themes Show Significant Correlation?", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["company"], fontsize=10)
    ax.legend(fontsize=10)
 
    # Add count labels on bars
    for i, row in summary_df.iterrows():
        ax.text(i - width / 2, row["pct_sig_close"] + 1, f"{row['sig_close']}/{row['total_themes']}",
                ha="center", fontsize=9)
        ax.text(i + width / 2, row["pct_sig_return"] + 1, f"{row['sig_return']}/{row['total_themes']}",
                ha="center", fontsize=9)
 
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "q2_significance_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
 
 
# ============================================================
# MAIN
# ============================================================
 
def main():
    print("=" * 70)
    print("BD733 — Q2: Theme Impact Analysis")
    print("=" * 70)
 
    # Load data
    df = load_data(DATA_PATH)
 
    # Run correlations per company per theme
    results_df = run_theme_correlations(df)
 
    # Get top themes per company
    top_themes = get_top_themes(results_df, n=10)
 
    # Print results
    print_results(results_df, top_themes)
    print_cross_company_comparison(results_df)
 
    # Save results to CSV (for use in Streamlit or report)
    results_path = os.path.join(OUTPUT_DIR, "q2_theme_correlation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nFull results saved to: {results_path}")
 
    # Save top themes separately (easier for Streamlit to consume)
    for company, top in top_themes.items():
        top_path = os.path.join(OUTPUT_DIR, f"q2_top_themes_{company.lower()}.csv")
        top.to_csv(top_path, index=False)
        print(f"Top themes for {company} saved to: {top_path}")
 
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_top_themes_per_company(top_themes)
    plot_significance_summary(results_df)
 
    print("\nDone! Check results/ folder for CSV and PNG outputs.")
 
 
if __name__ == "__main__":
    main()