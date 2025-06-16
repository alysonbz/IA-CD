# Importa a função para baixar o dataset e o módulo math
from ucimlrepo import fetch_ucirepo
import math

# Baixa o dataset Iris da UCI (ID 53) usando a biblioteca ucimlrepo
iris = fetch_ucirepo(id=53)
df = iris.data.original  # Obtém o DataFrame com os dados originais

# Converte o DataFrame para uma lista com valores numéricos
lista = []
for _, row in df.iterrows():
    features = [float(x) for x in row[:-1]]  # Converte as 4 primeiras colunas para float
    classe_str = row.iloc[-1]  # Obtém o nome da classe
    # Mapeia o nome da classe para números
    if classe_str == 'Iris-setosa':
        classe = 1.0
    elif classe_str == 'Iris-versicolor':
        classe = 2.0
    elif classe_str == 'Iris-virginica':
        classe = 3.0
    else:
        continue  # Ignora linhas inválidas
    lista.append(features + [classe])  # Junta as features com a classe e adiciona à lista

# Função para contar o número de amostras de cada classe
def countclasses(lista):
    setosa = versicolor = virginica = 0
    for amostra in lista:
        if amostra[4] == 1.0:
            setosa += 1
        elif amostra[4] == 2.0:
            versicolor += 1
        elif amostra[4] == 3.0:
            virginica += 1
    return setosa, versicolor, virginica

# Define a proporção de dados de treino
p = 0.6
# Conta quantas amostras existem de cada classe
setosa, versicolor, virginica = countclasses(lista)

# Separa os dados em treino e teste, mantendo a proporção de cada classe
treinamento, teste = [], []
max_setosa = int(p * setosa)
max_versicolor = int(p * versicolor)
max_virginica = int(p * virginica)
total1 = total2 = total3 = 0

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

# Funções de distância para o KNN

# Distância Euclidiana
def dist_euclidiana(v1, v2):
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1) - 1)))

# Distância Manhattan
def dist_manhattan(v1, v2):
    return sum(abs(v1[i] - v2[i]) for i in range(len(v1) - 1))

# Distância de Minkowski com parâmetro p
def dist_minkowski(v1, v2, p):
    return sum(abs(v1[i] - v2[i]) ** p for i in range(len(v1) - 1)) ** (1 / p)

# Distância de Chebyshev
def dist_chebyshev(v1, v2):
    return max(abs(v1[i] - v2[i]) for i in range(len(v1) - 1))

# Função do algoritmo KNN
def knn(treinamento, nova_amostra, K, distancia_func, **kwargs):
    dists = {}
    # Calcula a distância de cada amostra de treino até a nova amostra
    for i in range(len(treinamento)):
        # Usa argumentos extras se necessário (ex: p para Minkowski)
        d = distancia_func(treinamento[i], nova_amostra, **kwargs) if kwargs else distancia_func(treinamento[i], nova_amostra)
        dists[i] = d

    # Seleciona os K vizinhos mais próximos
    k_vizinhos = sorted(dists, key=dists.get)[:K]

    # Conta quantos vizinhos de cada classe existem
    qtd_setosa = qtd_versicolor = qtd_virginica = 0
    for indice in k_vizinhos:
        if treinamento[indice][-1] == 1.0:
            qtd_setosa += 1
        elif treinamento[indice][-1] == 2.0:
            qtd_versicolor += 1
        else:
            qtd_virginica += 1

    # Retorna a classe mais frequente entre os vizinhos
    return [qtd_setosa, qtd_versicolor, qtd_virginica].index(max([qtd_setosa, qtd_versicolor, qtd_virginica])) + 1.0

# Função para avaliar o desempenho do KNN com diferentes distâncias
def avaliar(nome, distancia_func, **kwargs):
    acertos = 0
    for amostra in teste:
        classe = knn(treinamento, amostra, K=1, distancia_func=distancia_func, **kwargs)
        if amostra[-1] == classe:
            acertos += 1
    print(f"Porcentagem de acertos ({nome}): {100 * acertos / len(teste):.2f}%")

# Avaliação do KNN com diferentes funções de distância
avaliar("Euclidiana", dist_euclidiana)
avaliar("Manhattan", dist_manhattan)
avaliar("Minkowski (p=3)", dist_minkowski, p=3)
avaliar("Chebyshev", dist_chebyshev)