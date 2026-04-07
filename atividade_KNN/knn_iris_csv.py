import math

lista = []

with open("iris_data.csv", "r") as f:
    for linha in f.readlines():
        valores = linha.strip().split(",")
        sublista = []

        for item in valores:
            sublista.append(float(item))

        lista.append(sublista)


def contar_classes(lista):
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

    return setosa, versicolor, virginica


p = 0.6
setosa, versicolor, virginica = contar_classes(lista)

treinamento = []
teste = []

max_setosa = int(p * setosa)
max_versicolor = int(p * versicolor)
max_virginica = int(p * virginica)

total1 = 0
total2 = 0
total3 = 0

for lis in lista:
    if lis[4] == 1.0 and total1 < max_setosa:
        treinamento.append(lis)
        total1 += 1
    elif lis[4] == 2.0 and total2 < max_versicolor:
        treinamento.append(lis)
        total2 += 1
    elif lis[4] == 3.0 and total3 < max_virginica:
        treinamento.append(lis)
        total3 += 1
    else:
        teste.append(lis)

# 4) FUNÇÕES DE DISTÂNCIA

# Aqui está a principal adaptação da atividade:
# além da euclidiana, também foram adicionadas
# manhattan, minkowski e chebyshev.

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

def calcular_distancia(v1, v2, tipo):
    if tipo == "euclidiana":
        return dist_euclidiana(v1, v2)
    elif tipo == "manhattan":
        return dist_manhattan(v1, v2)
    elif tipo == "minkowski":
        return dist_minkowski(v1, v2, 3)
    elif tipo == "chebyshev":
        return dist_chebyshev(v1, v2)
    else:
        return None

# IMPLEMENTAÇÃO DO KNN

# Essa é a parte principal:
# - calcula a distância da nova amostra até o treino
# - ordena as distâncias
# - escolhe os k vizinhos mais próximos
# - retorna a classe mais frequente

def knn(treinamento, nova_amostra, k, tipo_distancia):
    distancias = {}

    for i in range(len(treinamento)):
        d = calcular_distancia(treinamento[i], nova_amostra, tipo_distancia)
        distancias[i] = d

    k_vizinhos = sorted(distancias, key=distancias.get)[:k]

    qtd_setosa = 0
    qtd_versicolor = 0
    qtd_virginica = 0

    for indice in k_vizinhos:
        if treinamento[indice][4] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][4] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1

    contagem = [qtd_setosa, qtd_versicolor, qtd_virginica]
    return contagem.index(max(contagem)) + 1.0


k = 5
tipos_distancia = ["euclidiana", "manhattan", "minkowski", "chebyshev"]

for tipo in tipos_distancia:
    acertos = 0

    for amostra in teste:
        classe_prevista = knn(treinamento, amostra, k, tipo)

        if amostra[4] == classe_prevista:
            acertos += 1

    total_testes = len(teste)
    erros = total_testes - acertos
    porcentagem = 100 * acertos / total_testes

    print("Distância:", tipo)
    print("Acertos:", acertos)
    print("Erros:", erros)
    print("Total de testes:", total_testes)
    print("Porcentagem de acertos:", porcentagem)
    print("-" * 35)