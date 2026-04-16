import math

lista = []

with open('iris_data.csv', 'r') as f:
    for linha in f:
        linha = linha.strip()

        if not linha:
            continue

        a = linha.split(',')

        atributos = [float(x) for x in a[:4]]

        if a[4] == 'Iris-setosa':
            classe = 1.0
        elif a[4] == 'Iris-versicolor':
            classe = 2.0
        else:
            classe = 3.0

        lista.append(atributos + [classe])


def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0

    for i in range(len(lista)):
        if lista[i][4] == 1.0:
            setosa += 1
        elif lista[i][4] == 2.0:
            versicolor += 1
        elif lista[i][4] == 3.0:
            virginica += 1

    return [setosa, versicolor, virginica]


p = 0.6
setosa, versicolor, virginica = countclasses(lista)

treinamento, teste = [], []

max_setosa = int(p * setosa)
max_versicolor = int(p * versicolor)
max_virginica = int(p * virginica)

total1 = 0
total2 = 0
total3 = 0

for lis in lista:
    if lis[-1] == 1.0 and total1 < max_setosa:
        treinamento.append(lis)
        total1 += 1
    elif lis[-1] == 2.0 and total2 < max_versicolor:
        treinamento.append(lis)
        total2 += 1
    elif lis[-1] == 3.0 and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1
    else:
        teste.append(lis)


# ---------------- DISTANCIAS ----------------

def dist_euclidiana(v1, v2):
    soma = 0
    for i in range(len(v1) - 1):
        soma += (v1[i] - v2[i]) ** 2
    return math.sqrt(soma)


def dist_manhattan(v1, v2):
    soma = 0
    for i in range(len(v1) - 1):
        soma += abs(v1[i] - v2[i])
    return soma


def dist_minkowski(v1, v2, p=3):
    soma = 0
    for i in range(len(v1) - 1):
        soma += abs(v1[i] - v2[i]) ** p
    return soma ** (1 / p)


def dist_chebyshev(v1, v2):
    maior = 0
    for i in range(len(v1) - 1):
        diferenca = abs(v1[i] - v2[i])
        if diferenca > maior:
            maior = diferenca
    return maior


def calcular_distancia(v1, v2, tipo='euclidiana'):
    if tipo == 'euclidiana':
        return dist_euclidiana(v1, v2)
    elif tipo == 'manhattan':
        return dist_manhattan(v1, v2)
    elif tipo == 'minkowski':
        return dist_minkowski(v1, v2, p=3)
    elif tipo == 'chebyshev':
        return dist_chebyshev(v1, v2)
    else:
        raise ValueError('Tipo de distância inválido')


def knn(treinamento, nova_amostra, K, tipo_distancia='euclidiana'):
    dists = {}

    for i in range(len(treinamento)):
        d = calcular_distancia(treinamento[i], nova_amostra, tipo_distancia)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    qtd_setosa = 0
    qtd_versicolor = 0
    qtd_virginica = 0

    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1

    a = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) + 1.0



K = 3
tipos = ['euclidiana', 'manhattan', 'minkowski', 'chebyshev']

for tipo in tipos:
    acertos = 0

    for amostra in teste:
        classe = knn(treinamento, amostra, K, tipo)
        if amostra[-1] == classe:
            acertos += 1

    porcentagem = 100 * acertos / len(teste)
    print(f'Distância {tipo}: {porcentagem:.2f}% de acertos')