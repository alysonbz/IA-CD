import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# 1. CARREGAR DATASET
# =========================================================

df = pd.read_csv("datasets/housing.csv")

print(df.head())

# =========================================================
# 2. TRATAMENTO
# =========================================================

df.dropna(inplace=True)

df = pd.get_dummies(df, drop_first=True)

# =========================================================
# 3. DEFINIR X E Y
# =========================================================

y = df["median_house_value"]

X = df.drop("median_house_value", axis=1)

# =========================================================
# 4. TREINO E TESTE
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# 5. PADRONIZAÇÃO
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =========================================================
# 6. MODELOS
# =========================================================
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=1000)
}

# =========================================================
# 7. TREINAMENTO E MÉTRICAS
# =========================================================

resultados = {}

for nome, modelo in models.items():

    modelo.fit(X_train_scaled, y_train)

    pred = modelo.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, pred)

    mse = mean_squared_error(y_test, pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, pred)

    resultados[nome] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    print(f"\n===== {nome} =====")

    print("MAE:", mae)

    print("MSE:", mse)

    print("RMSE:", rmse)

    print("R²:", r2)

# =========================================================
# 8. REGRESSÃO LINEAR SIMPLES
# =========================================================

atributo = "median_income"

modelo_simples = LinearRegression()

modelo_simples.fit(df[[atributo]], y)

# GRÁFICO 1
plt.figure(figsize=(8,5))

plt.scatter(
    df[atributo],
    y,
    alpha=0.3
)

plt.plot(
    df[atributo],
    modelo_simples.predict(df[[atributo]]),
    color='red'
)

plt.xlabel("Median Income")

plt.ylabel("Median House Value")

plt.title("Regressão Linear Simples")

plt.savefig("regressao_linear_simples.png")

plt.show()

plt.close()

# =========================================================
# 9. REAL VS PREDITO
# =========================================================

modelo_final = Ridge(alpha=1.0)

modelo_final.fit(X_train_scaled, y_train)

pred_final = modelo_final.predict(X_test_scaled)

# GRÁFICO 2
plt.figure(figsize=(7,5))

plt.scatter(
    y_test,
    pred_final,
    alpha=0.5
)

plt.xlabel("Valores Reais")

plt.ylabel("Valores Preditos")

plt.title("Real vs Predito")

# Linha ideal
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red'
)

plt.savefig("real_vs_predito.png")

plt.show()

plt.close()

# =========================================================
# 10. VALIDAÇÃO CRUZADA
# =========================================================

scores_dict = {}

print("\n===== VALIDAÇÃO CRUZADA =====")

for nome, modelo in models.items():

    scores = cross_val_score(
        modelo,
        X_train_scaled,
        y_train,
        cv=5,
        scoring='r2'
    )

    scores_dict[nome] = scores

    print(f"\n{nome}")

    print("Scores:", scores)

    print("Média:", scores.mean())

# =========================================================
# 11. BOXPLOT DA VALIDAÇÃO CRUZADA
# =========================================================

# GRÁFICO 3
plt.figure(figsize=(8,5))

plt.boxplot(
    scores_dict.values(),
    labels=scores_dict.keys()
)

plt.ylabel("R²")

plt.title("Boxplot da Validação Cruzada")

plt.savefig("boxplot_validacao_cruzada.png")

plt.show()

plt.close()

# =========================================================
# 12. GRID SEARCH - RIDGE E LASSO
# =========================================================

from sklearn.model_selection import GridSearchCV

param_grid = {
    "alpha": [0.01, 0.1, 1, 10, 50, 100]
}

print("\n===== GRID SEARCH - RIDGE =====")

ridge_grid = GridSearchCV(
    Ridge(),
    param_grid,
    cv=5,
    scoring="r2"
)

ridge_grid.fit(X_train_scaled, y_train)

print("Melhor alpha Ridge:", ridge_grid.best_params_)
print("Melhor score Ridge:", ridge_grid.best_score_)


print("\n===== GRID SEARCH - LASSO =====")

lasso_grid = GridSearchCV(
    Lasso(max_iter=1000),
    param_grid,
    cv=5,
    scoring="r2"
)

lasso_grid.fit(X_train_scaled, y_train)

print("Melhor alpha Lasso:", lasso_grid.best_params_)
print("Melhor score Lasso:", lasso_grid.best_score_)