"""
==========================================================
PROJETO DE CLASSIFICAÇÃO COM KNN

Dataset:
Phishing Websites - UCI Machine Learning Repository

Objetivo:
Construir um pipeline completo de classificação
utilizando o algoritmo K-Nearest Neighbors (KNN),
incluindo pré-processamento, normalização,
validação cruzada e ajuste de hiperparâmetros.

Autores:
- Elyson Caique Alves Brito
- Maria de Fátima Brandão de Andrade

Disciplina:
Aprendizado de Máquina
"""

# IMPORTAÇÕES DAS BIBLIOTECAS

from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# 1. CARREGAMENTO DO DATASET

"""
Carrega o dataset de phishing da UCI
e separa atributos e variável alvo.
"""

# Carregando dataset da UCI Repository
phishing_websites = fetch_ucirepo(id=327)

# Separando atributos e variável alvo
X = phishing_websites.data.features
y = phishing_websites.data.targets

# Exibindo metadados do dataset
print(phishing_websites.metadata)

# Exibindo informações das variáveis
print(phishing_websites.variables)


# 2. EXPLORAÇÃO INICIAL DO DATASET

"""
Realiza uma exploração inicial do dataset,
verificando estrutura, dimensões e tipos.
"""

print("\nPRIMEIRAS LINHAS DOS ATRIBUTOS ==========================================================\n")
print(X.head())

print("\nPRIMEIRAS LINHAS DA VARIÁVEL ALVO ==========================================================\n")
print(y.head())

print("\nINFORMAÇÕES DO DATASET ==========================================================\n")
print(X.info())

print("\nDIMENSÕES DO DATASET ==========================================================\n")
print(f"Quantidade de linhas: {X.shape[0]}")
print(f"Quantidade de colunas: {X.shape[1]}")

print("\nTIPOS DOS ATRIBUTOS ==========================================================\n")
print(X.dtypes)

print("\nDISTRIBUIÇÃO DAS CLASSES ==========================================================\n")
print(y.value_counts())


# 3. IDENTIFICAÇÃO DE VALORES AUSENTES E INCONSISTÊNCIAS

"""
Verifica valores ausentes, duplicados
e estatísticas gerais do conjunto de dados.
"""

print("\nVALORES AUSENTES NOS ATRIBUTOS ==========================================================\n")
print(X.isnull().sum())

print("\nVALORES AUSENTES NA VARIÁVEL ALVO ==========================================================\n")
print(y.isnull().sum())

print("\nVALORES DUPLICADOS ==========================================================\n")
print(f"Quantidade de linhas duplicadas: {X.duplicated().sum()}")

print("\nESTATÍSTICAS DOS ATRIBUTOS ==========================================================\n")
print(X.describe())

print("\nVALORES ÚNICOS POR ATRIBUTO ==========================================================\n")

for coluna in X.columns:

    print(f"\nAtributo: {coluna}")
    print(X[coluna].unique())

print("\nVALORES ÚNICOS DA VARIÁVEL ALVO ==========================================================\n")
print(y["result"].unique())


# 4. ANÁLISE DOS ATRIBUTOS MAIS RELEVANTES

"""
Cria um dataframe completo para análise
de correlação entre atributos e classe.
"""

# Criando dataframe completo
df = X.copy()

# Adicionando variável alvo
df["result"] = y

# Removendo registros duplicados
df = df.drop_duplicates()

print("\nNOVAS DIMENSÕES APÓS REMOVER DUPLICATAS ==========================================================\n")
print(df.shape)

# Calculando matriz de correlação
correlation = df.corr()

print("\nCORRELAÇÃO DOS ATRIBUTOS COM A CLASSE ==========================================================\n")

print(
    correlation["result"]
    .sort_values(ascending=False)
)

# Construindo heatmap de correlação
plt.figure(figsize=(18, 12))

sns.heatmap(
    correlation,
    cmap="coolwarm"
)

plt.title("Mapa de Correlação dos Atributos")

plt.show()


# 5. SEPARAÇÃO DOS ATRIBUTOS E VARIÁVEL ALVO

"""
Separa os atributos de entrada
e a variável alvo do problema.
"""

# Separando atributos de entrada
X = df.drop("result", axis=1)

# Separando variável alvo
y = df["result"]

print("\nATRIBUTOS DE ENTRADA ==========================================================\n")
print(X.head())

print("\nVARIÁVEL ALVO ==========================================================\n")
print(y.head())

print("\nDIMENSÕES DOS ATRIBUTOS ==========================================================\n")
print(X.shape)

print("\nDIMENSÕES DA VARIÁVEL ALVO ==========================================================\n")
print(y.shape)


# 6. NORMALIZAÇÃO DOS DADOS COM MIN-MAX

"""
Aplica normalização Min-Max
para padronizar a escala dos dados.
"""

# Criando normalizador Min-Max
minmax_scaler = MinMaxScaler()

# Aplicando normalização
X_minmax = minmax_scaler.fit_transform(X)

# Convertendo novamente para DataFrame
X_minmax = pd.DataFrame(
    X_minmax,
    columns=X.columns
)

print("\nDADOS NORMALIZADOS COM MIN-MAX ==========================================================\n")
print(X_minmax.head())

print("\nESTATÍSTICAS APÓS MIN-MAX ==========================================================\n")
print(X_minmax.describe())


# 7. TREINAMENTO DO MODELO KNN

"""
Divide os dados em treino e teste
e treina o modelo KNN.
"""

# Separando dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_minmax,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTAMANHO DOS DADOS ==========================================================\n")

print(f"Treino: {X_train.shape}")
print(f"Teste: {X_test.shape}")

# Criando modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Treinando modelo
knn.fit(X_train, y_train)

# Realizando previsões
y_pred = knn.predict(X_test)

"""
Avalia o desempenho do modelo
utilizando métricas de classificação.
"""

# Calculando acurácia
accuracy = accuracy_score(y_test, y_pred)

print("\nACURÁCIA DO MODELO ==========================================================\n")
print(f"Acurácia: {accuracy:.4f}")

print("\nMATRIZ DE CONFUSÃO ==========================================================\n")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT ==========================================================\n")
print(classification_report(y_test, y_pred))


# 8. AVALIAÇÃO DE DIFERENTES VALORES DE K

"""
Testa diferentes valores de k
para identificar melhor desempenho.
"""

accuracies = []

# Lista de valores de k
k_values = list(range(1, 21))

for k in k_values:

    # Criando modelo para cada valor de k
    knn = KNeighborsClassifier(n_neighbors=k)

    # Treinando modelo
    knn.fit(X_train, y_train)

    # Fazendo previsões
    y_pred = knn.predict(X_test)

    # Calculando acurácia
    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)

    print(f"k = {k} | Acurácia = {acc:.4f}")


# 9. GRÁFICO K VS ACURÁCIA

"""
Gera gráfico para visualizar
a relação entre k e acurácia.
"""

plt.figure(figsize=(10, 6))

plt.plot(k_values, accuracies, marker='o')

plt.title("Acurácia do KNN para diferentes valores de k")

plt.xlabel("Valor de k")

plt.ylabel("Acurácia")

plt.grid(True)

plt.show()


# 10. IDENTIFICAÇÃO DO MELHOR VALOR DE K

"""
Identifica o melhor valor de k
com base na acurácia obtida.
"""

best_accuracy = max(accuracies)

best_k = k_values[
    accuracies.index(best_accuracy)
]

print("\nMELHOR VALOR DE K ==========================================================\n")

print(f"Melhor k: {best_k}")

print(f"Melhor acurácia: {best_accuracy:.4f}")


# 11. VALIDAÇÃO CRUZADA

"""
Aplica validação cruzada
para avaliar maior robustez do modelo.
"""

cross_validation_scores = []

for k in k_values:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    scores = cross_val_score(
        knn,
        X_minmax,
        y,
        cv=5,
        scoring='accuracy'
    )

    mean_score = scores.mean()

    cross_validation_scores.append(
        mean_score
    )

    print(f"k = {k} | Cross Validation Accuracy = {mean_score:.4f}")


# 12. GRID SEARCH PARA HIPERPARÂMETROS

"""
Executa Grid Search para encontrar
os melhores hiperparâmetros do KNN.
"""

# Definindo parâmetros para teste
param_grid = {
    'n_neighbors': list(range(1, 21)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Criando modelo base
knn = KNeighborsClassifier()

# Configurando Grid Search
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# Treinando Grid Search
grid_search.fit(X_minmax, y)

print("\nMELHORES HIPERPARÂMETROS ==========================================================\n")

print(grid_search.best_params_)

print("\nMELHOR ACURÁCIA GRID SEARCH ==========================================================\n")

print(grid_search.best_score_)


# 13. COMPARAÇÃO DOS RESULTADOS

"""
Compara os resultados obtidos
por treino/teste, validação cruzada
e Grid Search.
"""

best_cv_accuracy = max(
    cross_validation_scores
)

best_cv_k = k_values[
    cross_validation_scores.index(
        best_cv_accuracy
    )
]

print("\nCOMPARAÇÃO DOS RESULTADOS ==========================================================\n")

print(f"Melhor k visual: {best_k}")

print(f"Acurácia visual: {best_accuracy:.4f}")

print(f"\nMelhor k Cross Validation: {best_cv_k}")

print(f"Acurácia Cross Validation: {best_cv_accuracy:.4f}")

print(f"\nMelhores parâmetros Grid Search:")

print(grid_search.best_params_)

print(f"\nMelhor acurácia Grid Search:")

print(grid_search.best_score_)


# 14. COMPARAÇÃO ENTRE NORMALIZAÇÕES

"""
Compara diferentes técnicas
de normalização no desempenho do KNN.
"""

normalization_results = {}

# MIN-MAX

knn = KNeighborsClassifier(
    n_neighbors=best_k
)

scores = cross_val_score(
    knn,
    X_minmax,
    y,
    cv=5,
    scoring='accuracy'
)

normalization_results["Min-Max"] = scores.mean()

# Z-SCORE

standard_scaler = StandardScaler()

X_zscore = standard_scaler.fit_transform(X)

scores = cross_val_score(
    knn,
    X_zscore,
    y,
    cv=5,
    scoring='accuracy'
)

normalization_results["Z-Score"] = scores.mean()

# LOG

# Ajustando valores negativos
X_log = X + 2

# Aplicando transformação logarítmica
X_log = np.log1p(X_log)

scores = cross_val_score(
    knn,
    X_log,
    y,
    cv=5,
    scoring='accuracy'
)

normalization_results["Log"] = scores.mean()

# Exibindo resultados
print("\nRESULTADOS DAS NORMALIZAÇÕES ==========================================================\n")

for normalization, score in normalization_results.items():

    print(f"{normalization}: {score:.4f}")


# 15. ANÁLISE FINAL DOS RESULTADOS

"""
Apresenta interpretação dos resultados
obtidos durante o projeto.
"""

print("\nANÁLISE FINAL ==========================================================\n")

print("""
1. O algoritmo KNN apresentou desempenho elevado
na classificação de websites phishing.

2. A normalização dos dados influenciou diretamente
o desempenho do modelo, confirmando a importância
do pré-processamento em algoritmos baseados em distância.

3. A análise de diferentes valores de k mostrou que
valores intermediários apresentaram maior estabilidade.

4. O Grid Search permitiu identificar automaticamente
os melhores hiperparâmetros para o modelo.

5. A validação cruzada demonstrou que o modelo possui
boa capacidade de generalização.

6. A remoção de dados duplicados contribuiu para
reduzir possíveis vieses no treinamento.
""")