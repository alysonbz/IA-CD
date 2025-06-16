from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math

df = pd.read_csv('dataset/classificacao_ajustado.csv')

#1. Divida o dataset em treino e teste.
X = df.drop(['id','diagnosis'], axis=1)
y = df[['diagnosis']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Converte para numpy arrays
X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

# Monta listas com classe no final
treinamento = [list(X_train_np[i]) + [y_train_np[i]] for i in range(len(X_train_np))]
teste = [list(X_test_np[i]) + [y_test_np[i]] for i in range(len(X_test_np))]

#2. Implemente manualmente o KNN.
# Funções de distância

def dist_euclidiana(v1, v2):
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1) - 1)))

def dist_manhattan(v1, v2):
    return sum(abs(v1[i] - v2[i]) for i in range(len(v1) - 1))

def dist_chebyshev(v1, v2):
    return max(abs(v1[i] - v2[i]) for i in range(len(v1) - 1))

def dist_mahalanobis(v1, v2, inv_cov):
    diff = np.array(v1[:-1]) - np.array(v2[:-1])
    return math.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))

# Implementação do KNN manual

def knn(treinamento, nova_amostra, K, distancia_func, **kwargs):
    dists = {}
    for i, amostra in enumerate(treinamento):
        d = distancia_func(amostra, nova_amostra, **kwargs) if kwargs else distancia_func(amostra, nova_amostra)
        dists[i] = d
    k_vizinhos = sorted(dists, key=dists.get)[:K]
    votos = [treinamento[i][-1] for i in k_vizinhos]
    votos = [float(v.item()) if isinstance(v, np.ndarray) else float(v) for v in votos]  # <- conversão segura
    return max(set(votos), key=votos.count)

#3. Avalie usando: (Euclidiana, Manhattan, Chebyshev, Mahalanobis)

def avaliar(nome, distancia_func, **kwargs):
    acertos = 0
    for amostra in teste:
        classe_predita = knn(treinamento, amostra, K=3, distancia_func=distancia_func, **kwargs)
        if classe_predita == amostra[-1]:
            acertos += 1
    acuracia = 100 * acertos / len(teste)
    print(f"Acurácia com distância {nome}: {acuracia:.2f}%")

# Cálculo da matriz de covariância invertida para Mahalanobis
X_train_np = np.array([x[:-1] for x in treinamento])
cov = np.cov(X_train_np.T)
inv_cov = np.linalg.inv(cov)

#4. Compare os resultados obtidos para os diferentes valores de distancia, considerando a métrica acurácia.
avaliar("Euclidiana", dist_euclidiana)
avaliar("Manhattan", dist_manhattan)
avaliar("Chebyshev", dist_chebyshev)
avaliar("Mahalanobis", dist_mahalanobis, inv_cov=inv_cov)
