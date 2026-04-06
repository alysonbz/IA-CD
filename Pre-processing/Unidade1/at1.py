from ucimlrepo import fetch_ucirepo
import pandas as pd
import math

iris = fetch_ucirepo(id=53)
x = iris.data.features
y = iris.data.targets

df = pd.concat([x, y], axis=1)
df['class'] = df['class'].map({'Iris-setosa': 1.0, 'Iris-versicolor': 2.0, 'Iris-virginica': 3.0})
lista = df.values.tolist()

def countclasses(lista):
    setosa, versicolor, virginica = 0, 0, 0
    for i in range(len(lista)):
        if lista[i][4] == 1.0:
            setosa += 1
        if lista[i][4] == 2.0:
            versicolor += 1
        if lista[i][4] == 3.0:
            virginica += 1
    return [setosa, versicolor, virginica]

p = 0.6
setosa, versicolor, virginica = countclasses(lista)
treinamento, teste = [], []
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)
total1, total2, total3 = 0, 0, 0

for lis in lista:
    if lis[-1] == 1.0 and total1 < max_setosa:
        treinamento.append(lis)
        total1 += 1
    elif lis[-1] == 2.0 and total2 < max_versicolor:
        treinamento.append(lis)
        total2 +=1
    elif lis[-1] == 3.0 and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1
    else:
        teste.append(lis)

def dist_euclidiana(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(soma)

def dist_manhattan(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += abs(v1[i] - v2[i])
    return soma

def dist_minkowski(v1, v2, p=3):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += math.pow(abs(v1[i] - v2[i]), p)
    return math.pow(soma, 1/p)

def dist_chebyshev(v1, v2):
    dim, maior =len(v1), 0
    for i in range(dim - 1):
        diff = abs(v1[i] - v2[i])
        if diff > maior:
            maior = diff
    return maior

def knn(treinamento, nova_amostra, K, distancia):
    dists, len_treino = {}, len(treinamento)
    for i in range(len_treino):
        d = distancia(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1
    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 1.0

K = 1
distancias = {
    'Euclidiana' : dist_euclidiana,
    'Manhattan' : dist_manhattan,
    'Chebyshev' : dist_chebyshev,
    'Minkowski' : dist_minkowski
}

for nome, distancia in distancias.items():
    acertos = 0
    for amostra in teste:
        classe = knn(treinamento, amostra, K, distancia)
        if amostra[-1] == classe:
            acertos += 1
    print(f'{nome}: {100*acertos/len(teste):.2f}% de acertos')





