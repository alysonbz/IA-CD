dados = []

with open('iris.data', 'r') as arquivo:
    for linha in arquivo:
        if linha.strip():
            partes = linha.strip().split(',')
            atributos = [float(valor) for valor in partes[:-1]]
            classe_str = partes[-1]

            if classe_str == 'Iris-setosa':
                classe = 1.0
            elif classe_str == 'Iris-versicolor':
                classe = 2.0
            elif classe_str == 'Iris-virginica':
                classe = 3.0
            else:
                continue

            dados.append(atributos + [classe])

def contar_classes(amostras):
    contagem = [0, 0, 0]
    for amostra in amostras:
        indice = int(amostra[-1]) - 1
        contagem[indice] += 1
    return contagem

proporcao_treino = 0.6
quant_setosa, quant_versicolor, quant_virginica = contar_classes(dados)

limites = [
    int(proporcao_treino * quant_setosa),
    int(proporcao_treino * quant_versicolor),
    int(proporcao_treino * quant_virginica)
]

treino, teste = [], []
contadores = [0, 0, 0]

for item in dados:
    classe = int(item[-1]) - 1
    if contadores[classe] < limites[classe]:
        treino.append(item)
        contadores[classe] += 1
    else:
        teste.append(item)

import math

def distancia_euclidiana(v1, v2):
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1) - 1)))

def distancia_manhattan(v1, v2):
    return math.sqrt(sum(abs(v1[i] - v2[i]) for i in range(len(v1) - 1)))

def distancia_minkowski(v1, v2, p):
    return (sum(abs(v1[i] - v2[i]) ** p for i in range(len(v1) - 1))) ** (1 / p)

def distancia_chebyshev(v1, v2):
    return max(abs(v1[i] - v2[i]) for i in range(len(v1) - 1))

def classificar_knn(base_treino, nova_instancia, k, func_dist, **kwargs):
    distancias = []

    for idx, exemplo in enumerate(base_treino):
        if kwargs:
            distancia = func_dist(exemplo, nova_instancia, **kwargs)
        else:
            distancia = func_dist(exemplo, nova_instancia)
        distancias.append((idx, distancia))

    vizinhos_proximos = sorted(distancias, key=lambda x: x[1])[:k]
    votos = [0, 0, 0]

    for vizinho in vizinhos_proximos:
        classe_votada = int(base_treino[vizinho[0]][-1]) - 1
        votos[classe_votada] += 1

    return float(votos.index(max(votos)) + 1)

def avaliar_knn(treino, teste, k, func_dist, **kwargs):
    acertos = sum(
        1 for amostra in teste
        if classificar_knn(treino, amostra, k, func_dist, **kwargs) == amostra[-1]
    )
    return 100 * acertos / len(teste)

k_valor = 1
print("Acurácia (Euclidiana): {:.2f}%".format(avaliar_knn(treino, teste, k_valor, distancia_euclidiana)))
print("Acurácia (Manhattan): {:.2f}%".format(avaliar_knn(treino, teste, k_valor, distancia_manhattan)))
print("Acurácia (Chebyshev): {:.2f}%".format(avaliar_knn(treino, teste, k_valor, distancia_chebyshev)))
print("Acurácia (Minkowski, p=3): {:.2f}%".format(avaliar_knn(treino, teste, k_valor, distancia_minkowski, p=3)))
