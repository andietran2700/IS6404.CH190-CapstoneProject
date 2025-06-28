import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from scipy.stats import zscore
import matplotlib.pyplot as plt

# === SMA ===
class SlimeMouldAlgorithm:
    def __init__(self, func, lb, ub, dim, population=30, iterations=40):
        self.func = func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.population = population
        self.iterations = iterations
        self.positions = np.random.uniform(self.lb, self.ub, (population, dim))
        self.fitness = np.full(population, np.inf)
        self.best_pos = None
        self.best_fit = np.inf

    def optimize(self):
        for t in range(self.iterations):
            for i in range(self.population):
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                fit = self.func(self.positions[i])
                self.fitness[i] = fit
                if fit < self.best_fit:
                    self.best_fit = fit
                    self.best_pos = self.positions[i].copy()

            sorted_idx = np.argsort(self.fitness)
            W = np.zeros(self.population)
            for i in range(self.population):
                W[sorted_idx[i]] = 1 + np.log10(
                    (self.fitness.max() - self.fitness[sorted_idx[i]] + 1e-8) /
                    (self.fitness.max() - self.fitness.min() + 1e-8) + 1
                )

            for i in range(self.population):
                r = np.random.rand(self.dim)
                p = np.tanh(abs(self.fitness[i] - self.best_fit))
                vb = 1
                vc = np.random.uniform(-vb, vb, self.dim)
                if np.random.rand() < 0.5:
                    self.positions[i] = self.best_pos + vb * r * (W[i] * self.positions[i] - self.best_pos)
                else:
                    self.positions[i] = self.best_pos - vc * r * (W[i] * self.positions[i] - self.best_pos)

        return self.best_pos, self.best_fit

# === Load data ===
df = pd.read_csv("/content/drive/MyDrive/Business_Analysis/Capstone/data/preprocessed-data.csv")
df.dropna(subset=['Price', 'Year', 'Km', 'Condition', 'Origin', 'Style', 'Manufacture', 'Color', 'Seat', 'Window'], inplace=True)
df = df[np.abs(zscore(df['Price'])) < 3]

# === Feature Engineering ===
df['Car_Age'] = 2025 - df['Year']
df['Km_per_Year'] = df['Km'] / df['Car_Age'].replace(0, 1)
df['Is_New'] = (df['Car_Age'] <= 1).astype(int)

features = ['Car_Age', 'Km', 'Km_per_Year', 'Is_New', 'Condition', 'Origin',
            'Style', 'Manufacture', 'Color', 'Seat', 'Window']
X = df[features]
y = df['Price']

X_encoded = X.copy()
for col in X_encoded.select_dtypes(include='object'):
    X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_sma, X_val_sma, y_train_sma, y_val_sma = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# === Objective function for SMA ===
def rf_objective(params):
    n_estimators = int(np.clip(round(params[0]), 50, 500))
    max_depth = int(np.clip(round(params[1]), 5, 50))
    min_samples_split = int(np.clip(round(params[2]), 2, 20))
    min_samples_leaf = int(np.clip(round(params[3]), 1, 20))

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sma, y_train_sma)
    preds = model.predict(X_val_sma)
    r2 = r2_score(y_val_sma, preds)
    return 1 - r2  # SMA minimizes

# === SMA bounds and run ===
lb = [50, 5, 2, 1]
ub = [500, 50, 20, 20]
sma = SlimeMouldAlgorithm(func=rf_objective, lb=lb, ub=ub, dim=4)
best_params, best_loss = sma.optimize()

print("Best parameters:")
print(f"n_estimators={int(best_params[0])}, max_depth={int(best_params[1])}, "
      f"min_samples_split={int(best_params[2])}, min_samples_leaf={int(best_params[3])}")
print(f"Validation R²: {1 - best_loss:.4f}")

# === Train final model ===
final_rf = RandomForestRegressor(
    n_estimators=int(best_params[0]),
    max_depth=int(best_params[1]),
    min_samples_split=int(best_params[2]),
    min_samples_leaf=int(best_params[3]),
    random_state=42,
    n_jobs=-1
)
final_rf.fit(X_train, y_train)
final_preds = final_rf.predict(X_test)
val_preds = final_rf.predict(X_val_sma)

# === Custom Metrics ===
def a10_metric(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true) <= 0.1)

def full_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    a10 = a10_metric(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    residuals = y_true - y_pred
    sigma2 = np.var(residuals)
    mle = -n / 2 * np.log(2 * np.pi * sigma2) - (1 / (2 * sigma2)) * np.sum(residuals**2)

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'A10': a10 * 100,
        'R²': r2,
        'MLE': mle
    }

val_results = full_metrics(y_val_sma, val_preds)
test_results = full_metrics(y_test, final_preds)

print("\nValidation Set Metrics:")
for k, v in val_results.items():
    print(f"{k}: {v:.4f}")
print("\nTest Set Metrics:")
for k, v in test_results.items():
    print(f"{k}: {v:.4f}")

# # === Bar plot (excluding MLE) ===
# labels = [k for k in val_results.keys() if k != 'MLE']
# val_scores_plot = [val_results[k] for k in labels]
# test_scores_plot = [test_results[k] for k in labels]

# x = np.arange(len(labels))
# width = 0.35

# fig, ax = plt.subplots(figsize=(10, 6))
# bars1 = ax.bar(x - width/2, val_scores_plot, width, label='Validation')
# bars2 = ax.bar(x + width/2, test_scores_plot, width, label='Test')

# ax.set_ylabel('Score')
# ax.set_title('Model Performance Metrics (SMA + RF)')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
# ax.grid(True, linestyle='--', alpha=0.5)

# for bar in bars1 + bars2:
#     height = bar.get_height()
#     ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
#                 xytext=(0, 3), textcoords="offset points",
#                 ha='center', va='bottom', fontsize=9)

# plt.tight_layout()
# plt.show()