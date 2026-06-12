import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12, 
    'axes.labelsize': 14, 
    'axes.titlesize': 16,
    'legend.fontsize': 11
})

file_path = "FINAL_TRAINED_CYP3A7_PREDICTIONS.csv"
df_preds = pd.read_csv(file_path)


metrics_list = []

for split_name, group in df_preds.groupby('dataset_assignment'):
    # Extract actual and predicted columns
    y_true = group['y']
    y_pred = group['predicted_pIC50']
    
    # Calculate key regression statistics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics_list.append({
        'Dataset Split': split_name,
        'Count': len(group),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R² Score': round(r2, 4)
    })

df_metrics = pd.DataFrame(metrics_list)

print("--- MODEL PERFORMANCE METRICS ---")
print(df_metrics.to_string(index=False))
print("-" * 50)

best_row = df_metrics.loc[df_metrics['R² Score'].idxmax()]
print(f"\n★ Best R² Score achieved by: {best_row['Dataset Split']}")
print(f"  └─ Count: {best_row['Count']} | MAE: {best_row['MAE']} | RMSE: {best_row['RMSE']} | R² Score: {best_row['R² Score']}\n")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.scatterplot(
    data=df_preds, 
    x='y', 
    y='predicted_pIC50', 
    hue='dataset_assignment', 
    alpha=0.6, 
    ax=axes[0],
    palette='viridis'
)

min_val = min(df_preds['y'].min(), df_preds['predicted_pIC50'].min())
max_val = max(df_preds['y'].max(), df_preds['predicted_pIC50'].max())
axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Fit')

axes[0].set_title('CYP3A7: Actual vs. Predicted pIC50')
axes[0].set_xlabel('Experimental pIC50')
axes[0].set_ylabel('Predicted pIC50')
axes[0].legend(loc='upper left')

sns.histplot(
    data=df_preds, 
    x='absolute_error', 
    hue='dataset_assignment', 
    kde=True, 
    multiple='stack', 
    ax=axes[1],
    palette='viridis'
)

axes[1].set_title('Distribution of Absolute Errors')
axes[1].set_xlabel('Absolute Error (|Actual - Predicted|)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('cyp3a7_model_performance.png', dpi=300)
plt.show()