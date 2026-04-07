import math
# ------------- Pré-Processamento -----------------

lista = []  # Cria uma lista vazia para armazenar todos os dados da planilha.
with open('iris_data.csv', 'r') as f:  # "open" abre o arquivo; "r" significa read (leitura); "as f" nomeia o arquivo criado de f; "with" garante que o arquivo  será fechado automaticamente após a conclusão
    for linha in f.readlines(): # O comando readlines() lê o arquivo inteiro e divide linha por linha. E o for vai percorrer cada uma.
        a = linha.replace('\n', ''). split(',') # Substitui o \n no final da linha por '' para limpar o texto. split(',') corta a frase toda vez que encontra uma vírgula. Ele transforma a frase em uma lista de pedaços.
        lista.append(a) # adiciona a sub-lista criada dentro da lista principal.

# ------------ Verificando o Balanciamento --------------

def countclasses(lista):
    setosa = 0
    versicolor = 0
    virginica = 0
    for i in range(len(lista)):
        if lista[i][4] == 1.0:  # lista[i]: acessa a linha atual. [4]: acessa a quinta coluna dessa linha nde fica o rótulo/nome da flor. == 1.0: O código assume que, em algum passo anterior, os nomes das flores foram trocados por números.
            setosa += 1   # += 1: Se a condição for verdadeira, ele adiciona 1 ao contador
        if lista[i][4] == 2.0:
            versicolor += 1
        if lista[i][4] == 3.0:
            virginica += 1
    return [setosa, versicolor, virginica]

# ---------- Divisão de Treino e Teste ------------

p = 0.6
setosa, versicolor, virginica = countclasses(lista)

treinamento, teste = [], []
max_setosa, max_versicolor, max_virginica = int(p*setosa), int(p*versicolor), int(p*virginica)

total1 = 0
total2 = 0
total3 = 0

for lis in lista:
    if lis[-1] == 1.0 and total1<max_setosa
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

# --------- Distância Euclidiana -----------

def dist_euclidiana(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim -1):
        soma += math.pow(vi[i], -v2[i], 2)

#--------------- Aplicando KNN -------------

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

acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1]==classe:
        acertos +=1
print("Porcentagem de acertos:",100*acertos/len(teste))