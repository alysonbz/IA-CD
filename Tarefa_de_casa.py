lista = []

# Lendo o arquivo e convertendo as colunas para float, ignorando a primeira linha (cabeçalho) e a última linha (se vazia)
with open('iris_data.csv', 'r') as f:
    linhas = f.readlines()

    # Ignorando a primeira linha (cabeçalho) e a última linha (se vazia ou inválida)
    for linha in linhas[1:-1]:  # Começa na segunda linha e vai até a penúltima
        partes = linha.replace('\n', '').split(',')
        
        # Convertendo as características para float, e a classe será tratada separadamente
        dados = list(map(float, partes[:-1]))  # Características numéricas (todas as colunas, exceto a última)
        
        # Tratando a última coluna (classe), que é texto, e mapeando para números
        classe = partes[-1]
        if classe == 'Iris-setosa':
            classe_num = 1.0
        elif classe == 'Iris-versicolor':
            classe_num = 2.0
        elif classe == 'Iris-virginica':
            classe_num = 3.0
        else:
            continue  # Ignora qualquer classe desconhecida
        
        dados.append(classe_num)  # Adicionando a classe numérica
        lista.append(dados)

# Função para contar o número de ocorrências de cada classe
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

# Dividindo os dados entre treinamento e teste
treinamento, teste = [], []
max_setosa, max_versicolor, max_virginica = int(p * setosa), int(p * versicolor), int(p * virginica)
total1, total2, total3 = 0, 0, 0

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

# Função para calcular a distância euclidiana
import math
def dist_euclidiana(v1, v2):
    dim, soma = len(v1), 0
    for i in range(dim - 1):
        soma += math.pow(v1[i] - v2[i], 2)
    return math.sqrt(soma)

# Função KNN para classificação
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

# Avaliação do modelo
acertos, K = 0, 1
for amostra in teste:
    classe = knn(treinamento, amostra, K)
    if amostra[-1] == classe:
        acertos += 1

# Exibindo a porcentagem de acertos
print("Porcentagem de acertos:", 100 * acertos / len(teste))
