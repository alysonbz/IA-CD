# 1. Divida o dataset em treino e teste.

from sklearn.model_selection import train_test_split


X = df.drop("Outcome", axis=1) # Separação das colunas -Outcome.
y = df["Outcome"]


X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Divisão de treino e teste.

knn = KNeighborsClassifier(n_neighbors=5) # 17, 18
knn.fit(X_train, y_train) # Treinamento do modelo usando os dados de treino.

print(knn.score(X_test, y_test)) # Avaliação do modelo.

##

# 2. Implemente manualmente o KNN.

import numpy as np

# Features (características)/ Alvos (resultados), (1 (sim), 0 (não)).
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Conversão para float.
X_train = X_train.astype(float) 
X_test = X_test.astype(float)

# Distância euclidiana entre dois pontos.
def dis_euclidiana(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN manual
def knn_manual(X_train, y_train, x_teste, k=5):
    # Calculo da distancia do ponto teste para todos de treino.
    dist = [dis_euclidiana(x_teste, x_treino) for x_treino in X_train]
    index_neigh = np.argsort(dist)[:k] # Ordena as distancias de idx k vizinhos mais próximos.
    neigh_labels = [y_train[i] for i in index_neigh] # Pega o label de k 
    mais_comum = Counter(neigh_labels).most_common(1)[0][0] # Contagem do resultado que mais aparece entre os vizinhos e retorna a previsão.
    return mais_comum

# Previsões
y_predict = []
for x in X_test: # Para cada teste, realiza a previsão com KNN
    y_predict.append(knn_manual(X_train, y_train, x, k=18))

# Calculo da acurácia
acuracia = np.mean(np.array(y_predict) == y_test)
print(f"Acurácia do KNN manual : {acuracia:.4f}")

##

# 3. Avalie usando:
# Euclidiana
# Manhattan
# Chebyshev
# Mahalanobis

# Função de distância euclidiana
def dist_euclidiana(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Função de distância manhattan
def dist_manhattan(x1, x2):
    return np.sum(np.abs(x1 - x2))

# Função de distância chebyshev
def dist_chebyshev(x1, x2):
    return np.max(np.abs(x1 - x2))

# Função de distância mahalanobis
def dist_mahalanobis(x1, x2, VI):
    diff = x1 - x2
    return np.sqrt(np.dot(np.dot(diff.T, VI), diff))

#Previsão do KNN 
def knn_predict(X_train, y_train, X_test, k=5, metric="euclidean"):
    y_pred = []
    # Para Mahalanobis calcula a inversa da matriz de covariância dos dados de treino.
    if metric == "mahalanobis":
        VI = np.linalg.inv(np.cov(X_train.T))
    else:
        VI = None
    # Para cada teste:
    for x_teste in X_test:
        distancias = []
        for x_train in X_train: # Calcula a distancia de teste e todos os exemoplos de treino.
            if metric == "euclidean":
                d = dist_euclidiana(x_teste, x_train)
            elif metric == "manhattan":
                d = dist_manhattan(x_teste, x_train)
            elif metric == "chebyshev":
                d = dist_chebyshev(x_teste, x_train)
            elif metric == "mahalanobis":
                d = dist_mahalanobis(x_teste, x_train, VI)
            else:
                raise ValueError(f"Métrica desconhecida: {metric}")
            distancias.append(d)
        
        idxs = np.argsort(distancias)[:k] # Seleciona os idx de k vizinhos mais próximos.
        vizinhos = [y_train[i] for i in idxs] # Pega os rótulos de k
        voto_mais_comum = Counter(vizinhos).most_common(1)[0][0]
        y_pred.append(voto_mais_comum)
    return y_pred
#  Teste do KNN com as 4 métricas e retorna a acurácia de cada uma.
for dist_metric in ["euclidean", "manhattan", "chebyshev", "mahalanobis"]:
    print(f"\nDistância: {dist_metric}")
    y_pred = knn_predict(X_train, y_train, X_test, k=18, metric=dist_metric)
    acc = accuracy_score(y_test, y_pred)
    print("Acurácia:", round(acc, 4))
