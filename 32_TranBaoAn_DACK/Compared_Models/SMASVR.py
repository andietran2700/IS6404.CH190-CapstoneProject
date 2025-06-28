import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures


# === Slime Mould Algorithm ===
class SlimeMouldAlgorithm:
    def __init__(self, func, lb, ub, dim, population=50, iterations=100):
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
                W[sorted_idx[i]] = 1 + np.log10((self.fitness.max() - self.fitness[sorted_idx[i]] + 1e-8) /
                                               (self.fitness.max() - self.fitness.min() + 1e-8) + 1)

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


# === Load and preprocess data ===
df = pd.read_csv("/content/drive/MyDrive/Business_Analysis/Capstone/data/preprocessed-data.csv")
df.dropna(subset=['Price', 'Year', 'Km', 'Condition', 'Origin', 'Style', 'Manufacture', 'Color', 'Seat', 'Window'], inplace=True)
z_scores = np.abs(zscore(df['Price']))
df = df[z_scores < 3]

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
def svr_objective(params):
    C = 10 ** params[0]
    epsilon = 10 ** params[1]
    gamma = 10 ** params[2]
    model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X_train_sma, y_train_sma)
    preds = model.predict(X_val_sma)
    r2 = r2_score(y_val_sma, preds)
    return 1 - r2

lb = [-2, -4, -5]
ub = [4, 0, 1]

sma = SlimeMouldAlgorithm(func=svr_objective, lb=lb, ub=ub, dim=3, population=50, iterations=100)
best_params_log, best_loss = sma.optimize()

C = 10 ** best_params_log[0]
epsilon = 10 ** best_params_log[1]
gamma = 10 ** best_params_log[2]

print("Best SVR parameters:")
print(f"C={C:.5f}, epsilon={epsilon:.5f}, gamma={gamma:.5f}")
print(f"Validation R²: {1 - best_loss:.4f}")

# === Train final SVR model
final_svr = SVR(C=C, epsilon=epsilon, gamma=gamma)
final_svr.fit(X_train, y_train)
val_preds = final_svr.predict(X_val_sma)
test_preds = final_svr.predict(X_test)


# === Evaluation Metrics ===
def evaluate(y_true, y_pred):
    errors = y_true - y_pred
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(errors / y_true)) * 100
    a10 = np.mean(np.abs(errors) < 0.1 * y_true) * 100
    r2 = r2_score(y_true, y_pred)
    mle = np.sum(np.log(2 * np.pi * mse) / 2 + (errors ** 2) / (2 * mse))  # Log-likelihood assuming Gaussian errors

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "A10 (%)": a10,
        "R²": r2,
        "MLE": mle
    }


val_metrics = evaluate(y_val_sma, val_preds)
test_metrics = evaluate(y_test, test_preds)

print("\nValidation Set Performance:")
for k, v in val_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTest Set Performance:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")


# # === Bar Plot for Metrics ===
# labels = list(val_metrics.keys())
# x = np.arange(len(labels))
# width = 0.35

# fig, axs = plt.subplots(1, 1, figsize=(12, 6))
# axs.bar(x - width/2, [val_metrics[k] for k in labels], width, label='Validation', color='skyblue')
# axs.bar(x + width/2, [test_metrics[k] for k in labels], width, label='Test', color='salmon')

# axs.set_title('SMA-Optimized SVR Performance Metrics')
# axs.set_xticks(x)
# axs.set_xticklabels(labels, rotation=45)
# axs.legend()
# axs.grid(True, linestyle='--', alpha=0.6)

# plt.tight_layout()
plt.show()