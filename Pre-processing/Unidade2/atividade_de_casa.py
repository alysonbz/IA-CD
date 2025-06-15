
#importando o iris dataset através do sklearn

from sklearn.datasets import load_iris
iris = load_iris()

print(iris)
print("_______________________________________________________")


#Criando uma lista vazia e logo em seguida fazendo um for para colocar os elementos do dataset nessa lista e tranformando os elementos em float
lista = []

for i in range(len(iris.data)):
    linha = list(map(float, iris.data[i]))
    linha.append(float(iris.target[i]))

    #linha = list(map(str, iris.data[i]))
    #linha.append(str(iris.target[i]))
    lista.append(linha)

for l in lista[:5]:
    print(l)
print("_______________________________________________________")

#Fazendo uma função onde ele vai separar as classes sendo 0.0 a setosa, 0.1 a versicolor e 0.2 a virginica e depois faço um print para saber quantas há em cada uma
def countclasses(lista):
    setosa=0
    versicolor=0
    virginica=0
    for i in range(len(lista)):
        if lista[i][4] == 0.0:
            setosa += 1
        if lista[i][4] == 1.0:
            versicolor += 1
        if lista[i][4] == 2.0:
            virginica += 1

    return [setosa,versicolor,virginica]

setosa, versicolor, virginica = countclasses(lista)

print("Setosa: ", setosa, "Versicolor: ", versicolor,"Virginica: ", virginica)
print("_______________________________________________________")


#Aqui defino que 60% das minhas amostras vão para treino e separo em cada lista uma pra treino e outra para teste
p=0.6
setosa,versicolor, virginica = countclasses(lista)
treinamento, teste= [], []
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)
total1 =0
total2 =0
total3 =0
for lis in lista:
    if lis[-1]==0.0 and total1< max_setosa:
        treinamento.append(lis)
        total1 +=1
    elif lis[-1]==1.0 and total2<max_versicolor:
        treinamento.append(lis)
        total2 +=1
    elif lis[-1]==2.0 and total3<max_virginica:
        treinamento.append(lis)
        total3 +=1
    else:
        teste.append(lis)




#Importando a biblioteca math para realizar a operação da distância euclidiana
import math
def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)

def knn(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d= dist_euclidiana(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key= dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] ==0.0:
            qtd_setosa +=1
        elif treinamento[indice][-1] ==1.0:
            qtd_versicolor +=1
        else:
            qtd_virginica +=1
    a=[qtd_setosa, qtd_versicolor, qtd_virginica]
    #return a.index(max(a)) +1.0
    return float(a.index(max(a)))

acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))

#Distância minkowski
def minkowski(a, b, p=3):
    return np.sum(np.abs(a - b) ** p) ** (1 / p)

def knn_predict(X_train, y_train, x_test, k, p):
    distances = [minkowski(x_test, x_train, p) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]

import numpy as np
from collections import Counter
X = iris.data[:100, :2]
y = iris.target[:100]

x_novo = np.array([5.0, 3.5])
classe = knn_predict(X, y, x_novo, k=5, p=3)
print(f"Classe prevista: {classe}")

# Distância Chebyshev
def chebyshev(a, b):
    return np.max(np.abs(a - b))

def knn_predict(X_train, y_train, x_test, k):
    distances = [chebyshev(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]

x_novo = np.array([5.0, 3.5])
classe = knn_predict(X, y, x_novo, k=5)
print(f"Classe prevista: {classe}")

# Distância Manhattan
def manhattan(a, b):
    return np.sum(np.abs(a - b))

def knn_predict(X_train, y_train, x_test, k):
    distances = [manhattan(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]

x_novo = np.array([5.0, 3.5])
classe = knn_predict(X, y, x_novo, k=5)
print(f"Classe prevista: {classe}")

# Função de distância
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_test, k):
    distances = [euclidean(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]

x_novo = np.array([5.0, 3.5])
classe_predita = knn_predict(X, y, x_novo, k=5)
print(f"Classe prevista: {classe_predita}")

