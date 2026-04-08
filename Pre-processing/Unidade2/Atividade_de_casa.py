lista = []
class_mapping = {'Iris-setosa': 1.0, 'Iris-versicolor': 2.0, 'Iris-virginica': 3.0}

with open('iris-1.csv', 'r') as f:
    for linha in f:
        linha = linha.strip()
        if not linha:
            continue

        partes = linha.split(',')
        if len(partes) != 5:
            continue

        try:
            features = [float(p) for p in partes[:-1]]
            class_label_num = class_mapping[partes[-1].strip()]
            lista.append(features + [class_label_num])
        except (ValueError, KeyError):
            continue

def countclasses(lista):
    setosa=0
    versicolor=0
    virginica=0
    for i in range(len(lista)):
        if lista[i][4] == 1.0:
            setosa += 1
        if lista[i][4] == 2.0:
            versicolor += 1
        if lista[i][4] == 3.0:
            virginica += 1

    return [setosa,versicolor,virginica]

p=0.6
setosa,versicolor, virginica = countclasses(lista)
treinamento, teste= [], []
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)
total1 =0
total2 =0
total3 =0
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
        if treinamento[indice][-1] ==1.0:
            qtd_setosa +=1
        elif treinamento[indice][-1] ==2.0:
            qtd_versicolor +=1
        else:
            qtd_virginica +=1
    a=[qtd_setosa, qtd_versicolor, qtd_virginica]
    return a.index(max(a)) +1.0

acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))

import math

def dist_manhattan(v1, v2):
    dim, soma_abs = len(v1), 0
    for i in range(dim - 1):
        soma_abs += abs(v1[i] - v2[i])
    return soma_abs

def dist_chebyshev(v1, v2):
    dim, max_diff = len(v1), 0
    for i in range(dim - 1):
        diff = abs(v1[i] - v2[i])
        if diff > max_diff:
            max_diff = diff
    return max_diff

def dist_minkowski(v1, v2, p):
    if p < 1:
        raise ValueError("O parâmetro 'p' para a distância de Minkowski deve ser >= 1.")

    dim, soma_pot = len(v1), 0
    for i in range(dim - 1):
        soma_pot += math.pow(abs(v1[i] - v2[i]), p)
    return math.pow(soma_pot, 1/p)

def knn_generalized(treinamento, nova_amostra, K, dist_func, p=None):
    dists, len_treino = {}, len(treinamento)

    for i in range(len_treino):
        # Passa 'p' para a distância de Minkowski se for a função de distância selecionada
        if p is not None and dist_func.__name__ == 'dist_minkowski':
            d = dist_func(treinamento[i], nova_amostra, p)
        else:
            d = dist_func(treinamento[i], nova_amostra)
        dists[i] = d

    # Seleciona os K vizinhos mais próximos
    k_vizinhos = sorted(dists, key=dists.get)[:K]

    # Conta as ocorrências de cada classe entre os vizinhos
    qtd_setosa, qtd_versicolor, qtd_virginica = 0, 0, 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1

    # Retorna a classe majoritária. Se todas as contagens forem zero, retorna None.
    counts = [qtd_setosa, qtd_versicolor, qtd_virginica]
    if max(counts) == 0:
        return None

    return counts.index(max(counts)) + 1.0

K_value = 1 # Usando K=1 conforme a última execução

print(f"Comparando resultados para K={K_value} com diferentes métricas de distância:\n")

# --- Distância Euclidiana (reutilizando a função existente) ---
acertos_euclidean = 0
for amostra in teste:
    classe = knn_generalized(treinamento, amostra, K_value, dist_euclidiana)
    if amostra[-1] == classe:
        acertos_euclidean += 1
print(f"Porcentagem de acertos (Euclidiana): {100 * acertos_euclidean / len(teste):.2f}%")

# --- Distância Manhattan ---
acertos_manhattan = 0
for amostra in teste:
    classe = knn_generalized(treinamento, amostra, K_value, dist_manhattan)
    if amostra[-1] == classe:
        acertos_manhattan += 1
print(f"Porcentagem de acertos (Manhattan): {100 * acertos_manhattan / len(teste):.2f}%")

# --- Distância Chebyshev ---
acertos_chebyshev = 0
for amostra in teste:
    classe = knn_generalized(treinamento, amostra, K_value, dist_chebyshev)
    if amostra[-1] == classe:
        acertos_chebyshev += 1
print(f"Porcentagem de acertos (Chebyshev): {100 * acertos_chebyshev / len(teste):.2f}%")

# --- Distância Minkowski (com p=3 como exemplo) ---
acertos_minkowski = 0
for amostra in teste:
    classe = knn_generalized(treinamento, amostra, K_value, dist_minkowski, p=3)
    if amostra[-1] == classe:
        acertos_minkowski += 1
print(f"Porcentagem de acertos (Minkowski, p=3): {100 * acertos_minkowski / len(teste):.2f}%")
