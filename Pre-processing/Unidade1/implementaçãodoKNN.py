import math

lista = []
with open('iris_data.csv', 'r') as f:
    # f.readline() lê a primeira linha (cabeçalho) e a "descarta"
    header = f.readline()

    for linha in f.readlines():
        a = linha.replace('\n', '').split(',')

        # Ignoramos o índice 0 (Id) e pegamos do índice 1 ao 4 (os 4 atributos reais)
        # a[1:5] pega as posições 1, 2, 3 e 4
        b = list(map(float, a[1:5]))

        # A espécie agora está na posição 5
        if a[5] == 'Iris-setosa':
            classe = 1.0
        elif a[5] == 'Iris-versicolor':
            classe = 2.0
        else:
            classe = 3.0

        lista.append(b + [classe])


# --- O restante do seu código permanece igual, apenas verifique as funções abaixo ---

def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)):
        # A classe agora é o 5º elemento (índice 4)
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
max_setosa, max_versicolor, max_virginica = int(p * setosa), int(p * versicolor), int(p * virginica)
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


def dist_euclidiana(v1, v2):
    # dim-1 para não contar a classe no cálculo da distância
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(soma)


def knn(treinamento, nova_amostra, K):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        d = dist_euclidiana(treinamento[i], nova_amostra)
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


acertos, K = 0, 3  # Aumentei o K para 3 para melhor estabilidade
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1] == classe:
        acertos += 1

print("Porcentagem de acertos:", 100 * acertos / len(teste))