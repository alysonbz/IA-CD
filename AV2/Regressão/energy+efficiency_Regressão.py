# ==========================================================
# PROJETO DE REGRESSÃO
# Regressão Linear, Ridge e Lasso
# Dataset: Energy Efficiency (UCI)
# ==========================================================

# ==========================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split,cross_val_score,GridSearchCV)
from sklearn.linear_model import (LinearRegression,Ridge,Lasso)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error,r2_score)

# ==========================================================
# 2. CARREGAMENTO DO DATASET
# ==========================================================

energy_df = pd.read_excel('ENB2012_data.xlsx')

# Renomeando colunas
energy_df.columns = [
    "comp_relativa",
    "area_superf",
    "area_parede",
    "area_telhado",
    "altura_total",
    "orientacao",
    "area_vidro",
    "dist_vidro",
    "carga_aqueci",
    "carga_resfria"
]
print("\nPrimeiras linhas do dataset: \n", energy_df.head())

# ==========================================================
# 3. EXPLORAÇÃO DOS DADOS
# ==========================================================

print("\nInformações gerais: \n")
energy_df.info()

print("\nEstatísticas descritivas: \n", energy_df.describe())

# Valores ausentes
print("\nValores nulos: \n", energy_df.isnull().sum())

# Duplicados
print("\nQuantidade de duplicados: ", energy_df.duplicated().sum())

# ==========================================================
# 4. MATRIZ DE CORRELAÇÃO
# ==========================================================

correlacao = energy_df.corr(numeric_only=True)

plt.figure(figsize=(12, 8))

sns.heatmap(correlacao, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

plt.title("Mapa de Correlação - Energy Efficiency")
plt.xticks(rotation=45,ha="right",fontsize=8)
plt.yticks(rotation=0,ha="right",fontsize=8)
plt.show()

# ==========================================================
# 5. DEFINIÇÃO DE X E y
# ==========================================================

X = energy_df.drop(columns=["carga_aqueci", "carga_resfria"])
y = energy_df["carga_aqueci"]

# ==========================================================
# 6. DIVISÃO TREINO E TESTE
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================================
# 7. PRÉ-PROCESSAMENTO
# (necessário para Ridge e Lasso)
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# 8. REGRESSÃO LINEAR
# ==========================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

# Previsões do modelo
linear_pred = linear_model.predict(X_test)

# Comparação entre valores reais e previstos
resultado = pd.DataFrame({
    "Real": y_test.values,
    "Previsto": linear_pred
})
print("\nPrimeiras previsões do modelo:")
print(resultado.head())

linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))

linear_r2 = r2_score(y_test,linear_pred)

print("\n===== REGRESSÃO LINEAR =====")
print("RMSE:", linear_rmse)
print("R²:", linear_r2)

# ==========================================================
# 9. MODELO RIDGE
# ==========================================================

ridge_model = Ridge(alpha=0.1)

ridge_model.fit(X_train_scaled, y_train)

ridge_pred = ridge_model.predict(X_test_scaled)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))

ridge_r2 = r2_score(y_test, ridge_pred)

print("\n===== RIDGE =====")
print("RMSE:", ridge_rmse)
print("R²:", ridge_r2)

# ==========================================================
# 10. MODELO LASSO
# ==========================================================

lasso_model = Lasso(alpha=0.1)

lasso_model.fit(X_train_scaled, y_train)

lasso_pred = lasso_model.predict(X_test_scaled)

lasso_rmse = np.sqrt(mean_squared_error(y_test,lasso_pred))

lasso_r2 = r2_score(y_test, lasso_pred)

print("\n===== LASSO =====")
print("RMSE:", lasso_rmse)
print("R²:", lasso_r2)

# ==========================================================
# 11. COMPARAÇÃO DOS MODELOS
# ==========================================================

comparacao = pd.DataFrame({
    "Modelo": ["Linear", "Ridge", "Lasso"],
    "RMSE": [linear_rmse,ridge_rmse,lasso_rmse],
    "R²": [linear_r2,ridge_r2,lasso_r2]
})

print("\n===== COMPARAÇÃO DOS MODELOS =====")
print(comparacao)

# ==========================================================
# 12. GRÁFICO REAL VS PREVISTO
# ==========================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test,linear_pred)

plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color='red')

plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("Regressão Linear - Real vs Previsto")

plt.show()

# ==========================================================
# 13. ESCOLHA AUTOMÁTICA DO MELHOR ATRIBUTO
# PARA REGRESSÃO SIMPLES
# ==========================================================

correlacoes = correlacao["carga_aqueci"].drop(["carga_aqueci", "carga_resfria"])

melhor_atributo = correlacoes.abs().idxmax()

print("\nMelhor atributo encontrado:")
print(melhor_atributo)

# ==========================================================
# 14. REGRESSÃO LINEAR SIMPLES
# ==========================================================

X_simple = energy_df[[melhor_atributo]]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple,y,test_size=0.2,random_state=42)

simple_model = LinearRegression()

simple_model.fit(X_train_s, y_train_s)

simple_pred = simple_model.predict(X_test_s)

simple_rmse = np.sqrt(mean_squared_error(y_test_s, simple_pred))

simple_r2 = r2_score(y_test_s,simple_pred)

print("\n===== REGRESSÃO SIMPLES =====")
print("Atributo utilizado:", melhor_atributo)
print("RMSE:", simple_rmse)
print("R²:", simple_r2)

# ==========================================================
# 15. GRÁFICO DA RETA DE REGRESSÃO
# ==========================================================

ordem = X_test_s.iloc[:, 0].argsort()

plt.figure(figsize=(8, 6))

plt.scatter(X_test_s,y_test_s)

plt.plot(X_test_s.iloc[ordem], simple_pred[ordem])

plt.xlabel(melhor_atributo)
plt.ylabel("Carga de Aquecimento")
plt.title("Regressão Linear Simples")

plt.show()

# ==========================================================
# 16. VALIDAÇÃO CRUZADA
# ==========================================================

linear_cv = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")

ridge_cv = cross_val_score(Ridge(alpha=0.1), scaler.fit_transform(X), y, cv=5,scoring="r2")

lasso_cv = cross_val_score( Lasso(alpha=0.1), scaler.fit_transform(X), y, cv=5, scoring="r2")

print("\n===== VALIDAÇÃO CRUZADA =====")

print("\nLinear:")
print("R² médio:", linear_cv.mean())

print("\nRidge:")
print("R² médio:", ridge_cv.mean())

print("\nLasso:")
print("R² médio:", lasso_cv.mean())

# ==========================================================
# 17. GRID SEARCH
# ==========================================================

parametros = {"alpha": np.logspace(-4, 2, 20)}  # multiplição (escala logarítmica)

# Ridge
ridge_grid = GridSearchCV(Ridge(), parametros, cv=5, scoring="r2")

ridge_grid.fit(scaler.fit_transform(X), y)

# Lasso
lasso_grid = GridSearchCV(Lasso(max_iter=10000), parametros, cv=5, scoring="r2")

lasso_grid.fit(scaler.fit_transform(X), y)

print("\n===== GRID SEARCH =====")

print("\nRIDGE")
print("Melhor alpha:", ridge_grid.best_params_)
print("Melhor R²:", ridge_grid.best_score_)

print("\nLASSO")
print("Melhor alpha:", lasso_grid.best_params_)
print("Melhor R²:", lasso_grid.best_score_)

# ==========================================================
# 18. GRÁFICO DE IMPORTÂNCIA DOS ATRIBUTOS (LASSO)
# ==========================================================

lasso_importance = Lasso(alpha=0.01)

X_scaled = scaler.fit_transform(X)

lasso_importance.fit( X_scaled, y)

coeficientes = lasso_importance.coef_

plt.figure(figsize=(14,6))

plt.bar(X.columns, coeficientes)

plt.axhline(y=0, linestyle='--')

plt.xticks(rotation=30, ha='right')

plt.title("Importância das Variáveis - Lasso")
plt.ylabel("Coeficiente")
plt.xlabel("Variáveis")

plt.tight_layout()
plt.show()

# ==========================================================
# 19. COMPARAÇÃO FINAL
# ==========================================================

resultado_final = pd.DataFrame({
    "Modelo": ["Linear", "Ridge", "Lasso"],
    "R² Teste": [linear_r2, ridge_r2, lasso_r2],
    "R² Cross Validation": [linear_cv.mean(), ridge_cv.mean(), lasso_cv.mean()]
})

print("\n===== RESULTADO FINAL =====")
print(resultado_final)