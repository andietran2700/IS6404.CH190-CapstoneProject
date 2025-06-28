import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# --- Evaluation Functions ---
def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    y_pred_clipped = np.maximum(0, y_pred)
    msle = mean_squared_log_error(y_true, y_pred_clipped)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2,
        'MSLE': msle,
        'MAPE': mape
    }

def format_metrics(metrics_dict):
    return {
        k: int(v) if k == 'MSE' else round(v, 6)
        for k, v in metrics_dict.items()
    }

# --- Load and Clean Data ---
df = pd.read_csv("/content/drive/MyDrive/Business_Analysis/Capstone/data/preprocessed-data.csv")
df_clean = df.dropna(subset=[
    'Price', 'Year', 'Km', 'Condition', 'Origin',
    'Style', 'Manufacture', 'Color', 'Seat', 'Window'
])

features = ['Year', 'Km', 'Condition', 'Origin', 'Style',
            'Manufacture', 'Color', 'Seat', 'Window']
target = 'Price'
X = df_clean[features]
y = df_clean[target]

# Encode categorical features
X_encoded = X.copy()
for col in X_encoded.select_dtypes(include='object'):
    X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# --- Split into Train (60%), Validation (20%), Test (20%) ---
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

# --- Random Forest ---
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, criterion='squared_error',
                                 max_depth=5, min_samples_split=2,
                                 min_samples_leaf=2, bootstrap=True,
                                 oob_score=True, random_state=42,
                                 verbose=0, warm_start=True, ccp_alpha=1.0)
rf_model.fit(X_train, y_train)
preds_rf = rf_model.predict(X_val)
metrics_rf = evaluate_metrics(y_val, preds_rf)
formatted_rf = format_metrics(metrics_rf)

# --- SVR ---
print("\nTraining SVR...")
svr_model = SVR(kernel='rbf', degree=4, gamma='scale', coef0=0.0,
                tol=0.001, C=50, epsilon=0.2, cache_size=100, verbose=True)
svr_model.fit(X_train, y_train)
preds_svr = svr_model.predict(X_val)
metrics_svr = evaluate_metrics(y_val, preds_svr)
formatted_svr = format_metrics(metrics_svr)

# --- XGBoost ---
print("\nTraining XGBoost...")
xgb_model = XGBRegressor(learning_rate=0.005, n_estimators=100,
                         max_depth=4, random_state=42, alpha=0.5)
xgb_model.fit(X_train, y_train)
preds_xgb = xgb_model.predict(X_val)
metrics_xgb = evaluate_metrics(y_val, preds_xgb)
formatted_xgb = format_metrics(metrics_xgb)

# --- Print Validation Metrics ---
print("\nValidation Metrics:")
print(pd.DataFrame([formatted_rf], index=['Random Forest']).T)
print(pd.DataFrame([formatted_svr], index=['SVR']).T)
print(pd.DataFrame([formatted_xgb], index=['XGBoost']).T)

# --- Plotting Formatted Metrics ---
metrics_df = pd.DataFrame({
    'Random Forest': formatted_rf,
    'SVR': formatted_svr,
    'XGBoost': formatted_xgb
}).T

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.suptitle('Comparison of Regression Model Metrics (Validation Set)', fontsize=16)

def annotate_bars(ax, values, fmt="{:.0f}", fontsize=9):
    for i, val in enumerate(values):
        ax.text(i, val + max(values)*0.01, fmt.format(val), ha='center', va='bottom', fontsize=fontsize)

metrics_df[['MSE']].plot(kind='bar', ax=axes[0])
axes[0].set_title('MSE')
axes[0].set_ylabel('Value')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(loc='upper right')
axes[0].grid(axis='y')

metrics_df[['RMSE']].plot(kind='bar', ax=axes[1])
axes[1].set_title('RMSE')
axes[1].set_ylabel('Value')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(loc='upper right')
axes[1].grid(axis='y')

metrics_df[['MAE']].plot(kind='bar', ax=axes[2])
axes[2].set_title('MAE')
axes[2].set_ylabel('Value')
axes[2].tick_params(axis='x', rotation=0)
axes[2].legend(loc='upper right')
axes[2].grid(axis='y')

metrics_df[['R^2']].plot(kind='bar', ax=axes[3], color='orange')
axes[3].set_title('R-squared (Coefficient of Determination)')
axes[3].set_ylabel('Value')
axes[3].set_ylim(0, 1)
axes[3].tick_params(axis='x', rotation=0)
axes[3].legend(loc='upper right')
axes[3].grid(axis='y')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Evaluate Final Models on Test Set ---
print("\n--- Final Evaluation on Test Set ---")
final_preds_rf = rf_model.predict(X_test)
final_preds_svr = svr_model.predict(X_test)
final_preds_xgb = xgb_model.predict(X_test)

test_metrics_rf = format_metrics(evaluate_metrics(y_test, final_preds_rf))
test_metrics_svr = format_metrics(evaluate_metrics(y_test, final_preds_svr))
test_metrics_xgb = format_metrics(evaluate_metrics(y_test, final_preds_xgb))

print(pd.DataFrame([test_metrics_rf], index=['Random Forest - Test']).T)
print(pd.DataFrame([test_metrics_svr], index=['SVR - Test']).T)
print(pd.DataFrame([test_metrics_xgb], index=['XGBoost - Test']).T)

# --- Actual vs Predicted Plot on Test Set ---
results_df = pd.DataFrame({
    'Actual': y_test,
    'Random Forest': final_preds_rf,
    'SVR': final_preds_svr,
    'XGBoost': final_preds_xgb
})

plt.figure(figsize=(18, 6))
for i, model in enumerate(['Random Forest', 'SVR', 'XGBoost']):
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(x='Actual', y=model, data=results_df, alpha=0.6)
    slope, intercept = np.polyfit(results_df['Actual'], results_df[model], 1)
    x_vals = np.array([results_df['Actual'].min(), results_df['Actual'].max()])
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r--', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    plt.legend()
    plt.title(f"{model} Model\nActual vs Predicted (Test Set)")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

plt.tight_layout()
plt.show()