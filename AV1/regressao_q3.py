import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 0. Carregando dataset
df = pd.read_csv("regressao_ajustado.csv")
print(df.head(), '\n')
print(df.shape)

X = df["enginesize"].values.reshape(-1, 1)
y = df["price"].values

# 1. Aplique as regressões: Linear, Ridge e Lasso.
# Lista de modelos
modelos = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

# 2. Use `cross_val_score` com 5 folds.
# Armazenar resultados
resultados = []

for nome, modelo in modelos.items():
    # RMSE via cross_val_score
    neg_mse_scores = cross_val_score(modelo, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-neg_mse_scores)
    rmse_medio = rmse_scores.mean()

    # R² via cross_val_score
    r2_scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
    r2_medio = r2_scores.mean()

    # Salvar resultados
    resultados.append({
        'Modelo': nome,
        'RMSE Médio': rmse_medio,
        'R² Médio': r2_medio
    })

# 3. Compare RMSE e R² em uma tabela.
tabela_resultados = pd.DataFrame(resultados)
print("\nTabela Comparativa dos Modelos:\n")
print(tabela_resultados.sort_values(by="RMSE Médio"))

# 4. Identifique o melhor modelo.
melhor_modelo = tabela_resultados.sort_values(by="RMSE Médio").iloc[0]
print(f"\nMelhor modelo em termos de RMSE: {melhor_modelo['Modelo']}")