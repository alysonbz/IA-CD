import math

# Carregar os dados
lista = []
with open('iris.data', 'r') as f:
    for linha in f.readlines():
        if linha.strip():
            a = linha.replace('\n', '').split(',')
            lista.append(a)

# Função para contar classes
def countclasses(lista):
    setosa = versicolor = virginica = 0
    for item in lista:
        if item[4] == 'Iris-setosa':
            setosa += 1
        elif item[4] == 'Iris-versicolor':
            versicolor += 1
        elif item[4] == 'Iris-virginica':
            virginica += 1
    return [setosa, versicolor, virginica]

# Dividir em treino e teste
p = 0.6
setosa, versicolor, virginica = countclasses(lista)
treinamento, teste = [], []
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)
total1 = total2 = total3 = 0

for item in lista:
    if item[-1] == 'Iris-setosa' and total1 < max_setosa:
        treinamento.append(item)
        total1 += 1
    elif item[-1] == 'Iris-versicolor' and total2 < max_versicolor:
        treinamento.append(item)
        total2 += 1
    elif item[-1] == 'Iris-virginica' and total3 < max_virginica:
        treinamento.append(item)
        total3 += 1
    else:
        teste.append(item)

# Implementação das 4 distâncias
def euclidiana(v1, v2):
    soma = 0
    for i in range(len(v1)-1):
        soma += (float(v1[i]) - float(v2[i]))**2
    return math.sqrt(soma)

def manhattan(v1, v2):
    soma = 0
    for i in range(len(v1)-1):
        soma += abs(float(v1[i]) - float(v2[i]))
    return soma

def chebyshev(v1, v2):
    maior = 0
    for i in range(len(v1)-1):
        diff = abs(float(v1[i]) - float(v2[i]))
        if diff > maior:
            maior = diff
    return maior

def minkowski(v1, v2, p=3):
    soma = 0
    for i in range(len(v1)-1):
        soma += abs(float(v1[i]) - float(v2[i]))**p
    return soma**(1/p)

# Função KNN genérica que recebe a função de distância como parâmetro
def knn(treinamento, nova_amostra, K, distancia):
    dists = {}
    for i in range(len(treinamento)):
        d = distancia(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key=dists.get)[:K]

    contadores = {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}
    for idx in k_vizinhos:
        classe = treinamento[idx][-1]
        contadores[classe] += 1

    return max(contadores, key=contadores.get)

# Testando com as 4 distâncias
K = 1
distancias = {
    'Euclidiana': euclidiana,
    'Manhattan': manhattan,
    'Chebyshev': chebyshev,
    'Minkowski (p=3)': lambda x, y: minkowski(x, y, 3)
}

resultados = {}
for nome_dist, dist_func in distancias.items():
    acertos = 0
    for amostra in teste:
        predicao = knn(treinamento, amostra, K, dist_func)
        if predicao == amostra[-1]:
            acertos += 1
    taxa_acerto = 100 * acertos / len(teste)
    resultados[nome_dist] = taxa_acerto
    print(f"{nome_dist}: {taxa_acerto:.2f}% de acerto")

# Mostrar o melhor resultado
melhor_dist = max(resultados, key=resultados.get)
print(f"\nMelhor distância: {melhor_dist} com {resultados[melhor_dist]:.2f}% de acerto")