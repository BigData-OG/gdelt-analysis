import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def analyze_theme_importance(themes_data_path):

    df = pd.read_csv(themes_data_path)
    
    # Correlation by theme
    theme_correlations = df.groupby('theme_category').apply(
        lambda x: x['daily_theme_avg_tone'].corr(x['next_day_close'])
    ).sort_values(ascending=False)
    
    print("Top 20 themes correlated with next-day price:")
    print(theme_correlations.head(20))
    
    # 2. One-hot encode themes for ML
    theme_pivot = df.pivot_table(
        index=['event_date', 'ticker'],
        columns='theme_category',
        values='daily_theme_mentions',
        fill_value=0
    ).reset_index()
    
    # Merge with stock prices
    prices = df[['event_date', 'ticker', 'next_day_close']].drop_duplicates()
    merged = theme_pivot.merge(prices, on=['event_date', 'ticker'])
    
    # Train Random Forest to get feature importance
    feature_cols = [col for col in merged.columns if col not in ['event_date', 'ticker', 'next_day_close']]
    X = merged[feature_cols]
    y = merged['next_day_close']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'theme': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 themes by Random Forest importance:")
    print(importance_df.head(20))
    
    # Visualization
    plt.figure(figsize=(12, 8))
    importance_df.head(20).plot(x='theme', y='importance', kind='barh')
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Themes Affecting Stock Prices')
    plt.tight_layout()
    plt.savefig('theme_importance.png')
    
    return theme_correlations, importance_df