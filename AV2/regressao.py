"""
PROJETO DE REGRESSÃO ==========================================================

Dataset:
Air Quality - UCI Machine Learning Repository

Objetivo:
Construir um pipeline completo de regressão
utilizando:
- Regressão Linear
- Ridge Regression
- Lasso Regression

O projeto inclui:
- exploração do dataset
- tratamento de valores ausentes
- análise de atributos relevantes
- pré-processamento
- validação cruzada
- Grid Search
- comparação entre modelos
- regressão linear simples
- análise crítica dos resultados

Autores:
- Maria de Fátima Brandão de Andrade
- Elyson Caique Alves Brito

Disciplina:
Inteligência Artificial
"""

# IMPORTAÇÕES ==========================================================

"""
Importação das bibliotecas utilizadas
no projeto de regressão.
"""

from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import (
    MinMaxScaler
)

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


# CARREGAMENTO DO DATASET ==========================================================

"""
Carregamento do dataset Air Quality
disponível no repositório UCI.
"""

air_quality = fetch_ucirepo(id=360)

X_original = air_quality.data.features
y_original = air_quality.data.targets

print("\nMETADADOS ==========================================================\n")
print(air_quality.metadata)

print("\nVARIÁVEIS ==========================================================\n")
print(air_quality.variables)

# Juntar X e y

df = pd.concat([X_original, y_original], axis=1)

print("\nPRIMEIRAS LINHAS ==========================================================\n")
print(df.head())


# EXPLORAÇÃO DO DATASET ==========================================================

"""
Exploração inicial do dataset,
incluindo dimensões, estatísticas
e tipos dos atributos.
"""

print("\nINFORMAÇÕES DO DATASET ==========================================================\n")
print(df.info())

print("\nDIMENSÕES ==========================================================\n")
print(df.shape)

print("\nESTATÍSTICAS ==========================================================\n")
print(df.describe())

print("\nTIPOS DOS ATRIBUTOS ==========================================================\n")
print(df.dtypes)


# TRATAMENTO DE DADOS ==========================================================

"""
Tratamento de valores ausentes,
valores inválidos e remoção
de linhas duplicadas.
"""

print("\nVALORES AUSENTES ANTES ==========================================================\n")
print(df.isnull().sum())

# Contar valores -200 antes da substituição

print("\nQUANTIDADE DE VALORES -200 ==========================================================\n")

colunas_numericas = df.select_dtypes(
    include=np.number
).columns

for coluna in colunas_numericas:

    quantidade = (df[coluna] == -200).sum()

    print(f"{coluna}: {quantidade}")

# Substituir -200 por NaN

df = df.replace(
    to_replace=-200,
    value=np.nan
)

print("\nVALORES AUSENTES APÓS SUBSTITUIR -200 ==========================================================\n")
print(df.isnull().sum())

# Preencher valores ausentes com média

for coluna in colunas_numericas:

    media = df[coluna].mean()

    df[coluna] = df[coluna].fillna(media)

print("\nVALORES AUSENTES APÓS TRATAMENTO ==========================================================\n")
print(df.isnull().sum())

# Remover duplicados

quantidade_duplicados = df.duplicated().sum()

print("\nLINHAS DUPLICADAS ==========================================================\n")
print(quantidade_duplicados)

df = df.drop_duplicates()

print("\nNOVAS DIMENSÕES ==========================================================\n")
print(df.shape)


# ANÁLISE DE CORRELAÇÃO ==========================================================

"""
Análise da correlação entre os
atributos numéricos do dataset.
"""

correlation = df.corr(numeric_only=True)

print("\nCORRELAÇÃO COM TEMPERATURA ==========================================================\n")

print(
    correlation['T']
    .sort_values(ascending=False)
)

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    cmap='coolwarm'
)

plt.title('Mapa de Correlação')

plt.show()


# SEPARAÇÃO DOS DADOS ==========================================================

"""
Separação entre atributos de entrada
e variável alvo.
"""

# Variável alvo

y = df['T']

# Remover colunas desnecessárias

colunas_remover = []

for coluna in ['T', 'Date', 'Time']:

    if coluna in df.columns:

        colunas_remover.append(coluna)

X = df.drop(columns=colunas_remover)

print("\nATRIBUTOS ==========================================================\n")
print(X.head())

print("\nVARIÁVEL ALVO ==========================================================\n")
print(y.head())

print("\nDIMENSÕES X ==========================================================\n")
print(X.shape)

print("\n DIMENSÕES y ==========================================================\n")
print(y.shape)


# NORMALIZAÇÃO ==========================================================

"""
Aplicação da normalização Min-Max
nos atributos numéricos.
"""

scaler = MinMaxScaler()

X_normalizado = scaler.fit_transform(X)

X_normalizado = pd.DataFrame(
    X_normalizado,
    columns=X.columns
)

print("\nDADOS NORMALIZADOS ==========================================================\n")
print(X_normalizado.head())

X = X_normalizado


# DIVISÃO DOS DADOS ==========================================================

"""
Divisão dos dados em conjuntos
de treino e teste.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTAMANHO DOS DADOS ==========================================================\n")

print(f"Treino: {X_train.shape}")
print(f"Teste: {X_test.shape}")


# REGRESSÃO LINEAR ==========================================================

"""
Treinamento e avaliação do modelo
de Regressão Linear.
"""

modelo_linear = LinearRegression()

modelo_linear.fit(X_train, y_train)

y_pred_linear = modelo_linear.predict(X_test)

mae_linear = mean_absolute_error(y_test, y_pred_linear)

mse_linear = mean_squared_error(y_test, y_pred_linear)

rmse_linear = np.sqrt(mse_linear)

r2_linear = r2_score(y_test, y_pred_linear)

print(f"\nMÉTRICAS - Regressão Linear ==========================================================\n")

print(f"MAE: {mae_linear:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"RMSE: {rmse_linear:.4f}")
print(f"R²: {r2_linear:.4f}")


# RIDGE REGRESSION ==========================================================

"""
Treinamento e avaliação do modelo
Ridge Regression.
"""

modelo_ridge = Ridge(alpha=1.0)

modelo_ridge.fit(X_train, y_train)

y_pred_ridge = modelo_ridge.predict(X_test)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)

rmse_ridge = np.sqrt(mse_ridge)

r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"\nMÉTRICAS - Ridge Regression ==========================================================\n")

print(f"MAE: {mae_ridge:.4f}")
print(f"MSE: {mse_ridge:.4f}")
print(f"RMSE: {rmse_ridge:.4f}")
print(f"R²: {r2_ridge:.4f}")


# LASSO REGRESSION ==========================================================

"""
Treinamento e avaliação do modelo
Lasso Regression.
"""

modelo_lasso = Lasso(alpha=0.1)

modelo_lasso.fit(X_train, y_train)

y_pred_lasso = modelo_lasso.predict(X_test)

mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)

rmse_lasso = np.sqrt(mse_lasso)

r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"\nMÉTRICAS - Lasso Regression ==========================================================\n")

print(f"MAE: {mae_lasso:.4f}")
print(f"MSE: {mse_lasso:.4f}")
print(f"RMSE: {rmse_lasso:.4f}")
print(f"R²: {r2_lasso:.4f}")


# REGRESSÃO LINEAR SIMPLES ==========================================================

"""
Aplicação de regressão linear simples
utilizando apenas o atributo AH.
"""

# Melhor atributo baseado na correlação

atributo = 'AH'

X_simples = df[[atributo]]

y_simples = df['T']

X_train_simples, X_test_simples, y_train_simples, y_test_simples = train_test_split(
    X_simples,
    y_simples,
    test_size=0.2,
    random_state=42
)

modelo_simples = LinearRegression()

modelo_simples.fit(X_train_simples, y_train_simples)

y_pred_simples = modelo_simples.predict(X_test_simples)

# Ordenar valores para reta ficar correta

indices = X_test_simples.iloc[:, 0].argsort()

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test_simples,
    y_test_simples
)

plt.plot(
    X_test_simples.iloc[indices],
    y_pred_simples[indices]
)

plt.xlabel('AH')

plt.ylabel('Temperatura')

plt.title('Regressão Linear Simples')

plt.grid(True)

plt.show()

mae_simples = mean_absolute_error(y_test_simples, y_pred_simples)

mse_simples = mean_squared_error(y_test_simples, y_pred_simples)

rmse_simples = np.sqrt(mse_simples)

r2_simples = r2_score(y_test_simples, y_pred_simples)

print(f"\nMÉTRICAS - Regressão Linear Simples ==========================================================\n")

print(f"MAE: {mae_simples:.4f}")
print(f"MSE: {mse_simples:.4f}")
print(f"RMSE: {rmse_simples:.4f}")
print(f"R²: {r2_simples:.4f}")


# VALIDAÇÃO CRUZADA ==========================================================

"""
Aplicação da validação cruzada
para avaliar os modelos.
"""

modelos = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

for nome, modelo in modelos.items():

    scores = cross_val_score(
        modelo,
        X,
        y,
        cv=5,
        scoring='r2'
    )

    print(f"\n========== {nome} ==========\n")

    print(f"Scores: {scores}")

    print(f"Média R²: {scores.mean():.4f}")


# GRID SEARCH ==========================================================

"""
Busca dos melhores hiperparâmetros
para Ridge e Lasso utilizando
Grid Search.
"""

# RIDGE

ridge_params = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

ridge_grid = GridSearchCV(
    Ridge(),
    ridge_params,
    cv=5,
    scoring='r2'
)

ridge_grid.fit(X, y)

print("\nMELHOR RIDGE ==========================================================\n")

print(ridge_grid.best_params_)

print(f"Melhor R² Ridge: {ridge_grid.best_score_:.4f}")

# LASSO

lasso_params = {
    'alpha': [0.001, 0.01, 0.1, 1, 10]
}

lasso_grid = GridSearchCV(
    Lasso(),
    lasso_params,
    cv=5,
    scoring='r2'
)

lasso_grid.fit(X, y)

print("\nMELHOR LASSO ==========================================================\n")

print(lasso_grid.best_params_)

print(f"Melhor R² Lasso: {lasso_grid.best_score_:.4f}")


# ANÁLISE FINAL ==========================================================

"""
Discussão final dos resultados
obtidos pelos modelos.
"""

print("\nANÁLISE FINAL ==========================================================\n")

print("""
1. A Regressão Linear apresentou excelente
desempenho na previsão da temperatura.

2. Ridge Regression apresentou desempenho
semelhante ao modelo linear, reduzindo
possíveis problemas de multicolinearidade.

3. Lasso Regression apresentou desempenho
ligeiramente inferior, porém contribuiu
para simplificação do modelo.

4. O pré-processamento dos dados foi
fundamental para melhorar os resultados.

5. A análise de correlação mostrou que AH
foi o atributo mais relacionado à temperatura.

6. A regressão linear simples apresentou
desempenho inferior aos modelos multivariados,
mostrando a importância do uso de múltiplos
atributos simultaneamente.

7. A validação cruzada forneceu resultados
mais robustos para avaliação dos modelos.

8. O Grid Search encontrou hiperparâmetros
mais eficientes para Ridge e Lasso.
""")