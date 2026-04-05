import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def analyze_exposure_correlation(data_path):
    df = pd.read_csv(data_path)
    
    # pearson
    corr_price, p_price = pearsonr(df['daily_exposure_count'], df['Close'])
    corr_return, p_return = pearsonr(df['daily_exposure_count'], df['daily_return_pct'])
    
    print(f"Exposure vs. Close Price: r={corr_price:.3f}, p={p_price:.4f}")
    print(f"Exposure vs. Daily Return: r={corr_return:.3f}, p={p_return:.4f}")
    
    # Time-lagged correlation
    for lag in [1, 3, 7, 14]:
        df[f'return_{lag}d'] = df.groupby('ticker')['Close'].pct_change(lag)
        corr = df['daily_exposure_count'].corr(df[f'return_{lag}d'])
        print(f"{lag}-day lagged return correlation: {corr:.3f}")
    
    # Per-company analysis
    company_corr = df.groupby('company').apply(
        lambda x: x['daily_exposure_count'].corr(x['daily_return_pct'])
    )
    print("\nPer-company correlations:")
    print(company_corr.sort_values(ascending=False))
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(df['daily_exposure_count'], df['daily_return_pct'], alpha=0.3)
    plt.xlabel('Daily News Exposure Count')
    plt.ylabel('Daily Return %')
    plt.title(f'Exposure vs. Returns (r={corr_return:.3f})')
    plt.savefig('exposure_correlation.png')