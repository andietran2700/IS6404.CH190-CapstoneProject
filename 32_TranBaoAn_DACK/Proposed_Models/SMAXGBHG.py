import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import zscore
import xgboost as xgb

# === Metrics ===
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100

def mle(y_true, y_pred):
    residuals = y_true - y_pred
    sigma2 = np.var(residuals)
    n = len(y_true)
    return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

def a10_metric(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) < 0.1)

def evaluate_all(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "MLE": mle(y_true, y_pred),
        "A10": a10_metric(y_true, y_pred)
    }

# === HGR Loss ===
def heteroscedastic_gaussian_loss(y_true, y_pred):
    residual = y_pred - y_true
    grad = residual / (1.0 + residual**2)
    hess = (1 - residual**2) / (1.0 + residual**2)**2
    return grad, hess

# === SMA Optimizer ===
class SlimeMouldAlgorithm:
    def __init__(self, func, lb, ub, dim, population=50, iterations=80):
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
        self.history = []

    def optimize(self):
        for _ in range(self.iterations):
            for i in range(self.population):
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                fit = self.func(self.positions[i])
                self.fitness[i] = fit
                if fit < self.best_fit:
                    self.best_fit = fit
                    self.best_pos = self.positions[i].copy()
            self.history.append(self.best_fit)

            sorted_idx = np.argsort(self.fitness)
            W = np.zeros(self.population)
            for i in range(self.population):
                W[sorted_idx[i]] = 1 + np.log10((self.fitness.max() - self.fitness[sorted_idx[i]] + 1e-8) /
                                                (self.fitness.max() - self.fitness.min() + 1e-8) + 1)

            for i in range(self.population):
                r = np.random.rand(self.dim)
                vb = 1
                vc = np.random.uniform(-vb, vb, self.dim)
                if np.random.rand() < 0.5:
                    self.positions[i] = self.best_pos + vb * r * (W[i] * self.positions[i] - self.best_pos)
                else:
                    self.positions[i] = self.best_pos - vc * r * (W[i] * self.positions[i] - self.best_pos)

        return self.best_pos, self.best_fit

# === Load dataset ===
df = pd.read_csv("/content/drive/MyDrive/Business_Analysis/Capstone/data/preprocessed-data.csv")
df.dropna(subset=['Price', 'Year', 'Km', 'Condition', 'Origin', 'Style', 'Manufacture', 'Color', 'Seat', 'Window'], inplace=True)
z_scores = np.abs(zscore(df['Price']))
df = df[z_scores < 3]

df['Car_Age'] = 2025 - df['Year']
df['Km_per_Year'] = df['Km'] / df['Car_Age'].replace(0, 1)
df['Is_New'] = (df['Car_Age'] <= 1).astype(int)

features = ['Car_Age', 'Km', 'Km_per_Year', 'Is_New', 'Condition', 'Origin', 'Style', 'Manufacture', 'Color', 'Seat', 'Window']
X = df[features]
y = df['Price']

X_encoded = X.copy()
for col in X_encoded.select_dtypes(include='object'):
    X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_encoded)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# === Objective for SMA ===
def xgb_objective(params):
    max_depth = int(params[0])
    learning_rate = 10 ** params[1]
    n_estimators = int(params[2])
    subsample = params[3]
    colsample_bytree = params[4]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X_train):
        X_tr, X_va = X_train[train_index], X_train[val_index]
        y_tr, y_va = y_train.iloc[train_index], y_train.iloc[val_index]

        model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=heteroscedastic_gaussian_loss,
            verbosity=0,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        scores.append(r2_score(y_va, preds))

    return 1 - np.mean(scores)

# === SMA tuning ===
lb = [3, -3, 50, 0.5, 0.5]
ub = [10, 0, 300, 1.0, 1.0]
sma = SlimeMouldAlgorithm(func=xgb_objective, lb=lb, ub=ub, dim=5, population=50, iterations=80)
best_params, best_loss = sma.optimize()

max_depth = int(best_params[0])
learning_rate = 10 ** best_params[1]
n_estimators = int(best_params[2])
subsample = best_params[3]
colsample_bytree = best_params[4]

print("\nBest XGBoost Parameters:")
print(f"max_depth={max_depth}, learning_rate={learning_rate:.5f}, n_estimators={n_estimators}, subsample={subsample:.3f}, colsample_bytree={colsample_bytree:.3f}")
print(f"Validation R²: {1 - best_loss:.4f}")

# === Train final model and evaluate ===
final_model = xgb.XGBRegressor(
    max_depth=max_depth,
    learning_rate=learning_rate,
    n_estimators=n_estimators,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    objective=heteroscedastic_gaussian_loss,
    verbosity=0,
    n_jobs=-1,
    random_state=42
)

final_model.fit(X_train, y_train)
val_preds = final_model.predict(X_val)
test_preds = final_model.predict(X_test)

val_results = evaluate_all(y_val, val_preds)
test_results = evaluate_all(y_test, test_preds)

print("\nValidation Metrics:")
for metric, value in val_results.items():
    print(f"{metric}: {value:.4f}")

print("\nTest Metrics:")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

# === Plot Convergence ===
plt.figure(figsize=(8, 5))
plt.plot(sma.history, marker='o')
plt.title("SMA Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness (1 - R²)")
plt.grid(True)
plt.tight_layout()
plt.show()