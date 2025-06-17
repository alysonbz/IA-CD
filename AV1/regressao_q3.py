import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# 1. Carregar o dataset ajustado
df = pd.read_csv("regressao_ajustado.csv")

# Separar X e y
target = 'Y house price of unit area'
X = df.drop(columns=[target])
y = df[target]


# 2. Funções de avaliação (RMSE e R²)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_scorer = make_scorer(rmse, greater_is_better=False)  # Negativo para uso com cross_val_score

# 3. Modelos a comparar
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

# 4. Validação cruzada com 5 folds
results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # Calcular RMSE (negativo, então multiplicamos por -1)
    rmse_scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer)
    mean_rmse = -np.mean(rmse_scores)

    # Calcular R²
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    mean_r2 = np.mean(r2_scores)

    results.append({
        "Modelo": name,
        "RMSE médio": round(mean_rmse, 4),
        "R² médio": round(mean_r2, 4)
    })

# 5. Mostrar resultados em tabela
results_df = pd.DataFrame(results).sort_values(by="RMSE médio")
print("Comparação entre modelos:")
print(results_df.to_string(index=False))

# 6. Identificar melhor modelo
melhor = results_df.loc[results_df["RMSE médio"].idxmin()]
print(f"\n✅ Melhor modelo: {melhor['Modelo']} (RMSE = {melhor['RMSE médio']}, R² = {melhor['R² médio']})")
