import math


lista = []
## ERRO 1CONVERTER A ÚLTIMA COLUNA PARA 'FLOAT'
with open('iris_data.csv', 'r') as f:
    for linha in f.readlines():
        # separando os dados
        parts = linha.replace('\n','').split(',')

        # Converte os números para float
        numeric_features = list(map(float, parts[:-1]))

        # Mapear o nome da classe(string) para valores numéricos e converter a classe
        class_name = parts[-1]
        if class_name == 'Iris-setosa':
            numeric_features.append(1.0)
        elif class_name == 'Iris-versicolor':
            numeric_features.append(2.0)
        elif class_name == 'Iris-virginica':
            numeric_features.append(3.0)

        lista.append(numeric_features)

def countclasses(lista):
    setosa=0
    versicolor=0
    virginica=0
    for i in range(len(lista)):
        if lista[i] [4] == 1.0:
            setosa += 1
        if lista[i] [4] == 2.0:
            versicolor += 1
        if lista[i] [4] == 3.0:
            virginica += 1

    return [setosa, versicolor, virginica]

p =0.6
setosa,versicolor, virginica = countclasses(lista)

treinamento, teste= [], []

#max_setosa, max_versicolor, max_virginica = int(p*setosa),
#nt(p*versicolor), int(p*virginica)

#ERRO 2 - TUPLA COM 1 VALOR SÓ
max_setosa = int(p* setosa)
max_versicolor = int(p * versicolor)
max_virginica = int(p * virginica)


total1= 0
total2=0
total3=0

for lis in lista:
    if lis[-1]==1.0 and total1< max_setosa:
        treinamento.append(lis)
        total1 +=1
    elif lis[-1]==2.0 and total2<max_versicolor:
        treinamento.append(lis)
        total2 +=1
    elif lis[-1]==3.0 and total3<max_virginica:
        treinamento.append(lis)
        total3 +=1
    else:
        teste.append(lis)

def dist_euclidiana(v1,v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(v1[i] -v2[i],2)
    return math.sqrt(soma)

def dist_manhattan(v1, v2):
    soma = 0
    for i in range(len(v1)-1):
        soma += abs(v1[i] - v2[i])
    return soma

def dist_minkowsko(v1, v2, p=3):
    soma =0
    for i in range(len(v1) -1):
        soma += abs(v1[i] - v2[i]) **p
    return soma**(1/p)


def dist_chebyshev(v1, v2):
    maior = 0
    for i in range(len(v1)-1):
        d = abs(v1[i] - v2[i])
        if d > maior:
            maior = d
    return maior


def knn(treinamento, nova_amostra, K, func_dist):
    dists = {}

    for i in range(len(treinamento)):
        d= func_dist(treinamento[i], nova_amostra)
        dists[i] = d

    k_vizinhos = sorted(dists, key= dists.get)[:K]

    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] ==1.0:
            qtd_setosa +=1
        elif treinamento[indice][-1] ==2.0:
            qtd_versicolor +=1
        else:
            qtd_virginica +=1
    a=[qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) +1.0

K=3



#indentação das outras distâncias
for nome, func in [
    ("Euclidiana", dist_euclidiana),
    ("Manhattan", dist_manhattan),
    ("Minkowski", dist_minkowsko),
    ("Chebyshev", dist_chebyshev)
    ]:
        acertos = 0

        for amostra in teste:
            classe = knn(treinamento, amostra, K, func)
            if amostra[-1] == classe:
                acertos += 1
        print(nome, "->", 100*acertos/len(teste))