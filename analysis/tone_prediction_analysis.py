import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def evaluate_tone_prediction(predictions_df):
    
    # Calculate metrics
    r2 = r2_score(predictions_df['actual'], predictions_df['predicted'])
    rmse = np.sqrt(mean_squared_error(predictions_df['actual'], predictions_df['predicted']))
    mae = mean_absolute_error(predictions_df['actual'], predictions_df['predicted'])
    
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    
    # Direction accuracy (did we predict up/down correctly?)
    predictions_df['actual_direction'] = np.sign(predictions_df['actual'] - predictions_df['prev_close'])
    predictions_df['predicted_direction'] = np.sign(predictions_df['predicted'] - predictions_df['prev_close'])
    direction_accuracy = (predictions_df['actual_direction'] == predictions_df['predicted_direction']).mean()
    
    print(f"Direction Accuracy: {direction_accuracy:.2%}")
    
    # Tone-specific analysis
    # Bin tone into categories
    predictions_df['tone_category'] = pd.cut(
        predictions_df['daily_avg_tone'], 
        bins=[-100, -5, 5, 100],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    tone_performance = predictions_df.groupby('tone_category').apply(
        lambda x: r2_score(x['actual'], x['predicted'])
    )
    
    print("\nPrediction accuracy by tone category:")
    print(tone_performance)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Actual vs Predicted
    axes[0, 0].scatter(predictions_df['actual'], predictions_df['predicted'], alpha=0.3)
    axes[0, 0].plot([predictions_df['actual'].min(), predictions_df['actual'].max()],
                     [predictions_df['actual'].min(), predictions_df['actual'].max()], 'r--')
    axes[0, 0].set_xlabel('Actual Next-Day Close')
    axes[0, 0].set_ylabel('Predicted Next-Day Close')
    axes[0, 0].set_title(f'Predictions (R²={r2:.3f})')
    
    # Residuals
    residuals = predictions_df['actual'] - predictions_df['predicted']
    axes[0, 1].scatter(predictions_df['predicted'], residuals, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Value')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    
    # Tone vs Prediction Error
    axes[1, 0].scatter(predictions_df['daily_avg_tone'], np.abs(residuals), alpha=0.3)
    axes[1, 0].set_xlabel('Daily Average Tone')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Tone vs Prediction Error')
    
    # Direction accuracy over time
    predictions_df['date'] = pd.to_datetime(predictions_df['event_date'])
    monthly_accuracy = predictions_df.set_index('date').resample('M').apply(
        lambda x: (x['actual_direction'] == x['predicted_direction']).mean()
    )
    axes[1, 1].plot(monthly_accuracy.index, monthly_accuracy.values)
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Direction Accuracy')
    axes[1, 1].set_title('Direction Accuracy Over Time')
    
    plt.tight_layout()
    plt.savefig('tone_prediction_evaluation.png')