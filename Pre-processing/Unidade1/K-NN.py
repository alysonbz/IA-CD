# Distância Euclidiana =================================================================================================
def distancia_euclidiana(a, b):
    soma = 0

    for i in range(len(a)):
        diferenca = a[i] - b[i]
        soma = soma + (diferenca * diferenca)

    return soma ** 0.5
# Distância Manhattan ==================================================================================================
def distancia_manhattan(a, b):
    soma = 0

    for i in range(len(a)):
        diferenca = a[i] - b[i]

        if diferenca < 0:
            diferenca = -diferenca

        soma = soma + diferenca

    return soma
# Distância Minkowski ==================================================================================================
def distancia_minkowski(a, b, p):
    soma = 0

    for i in range(len(a)):
        diferenca = a[i] - b[i]

        if diferenca < 0:
            diferenca = -diferenca

        soma = soma + (diferenca ** p)

    return soma ** (1 / p)
# Distância Chebyshev ==================================================================================================
def distancia_chebyshev(a, b):
    maior = 0

    for i in range(len(a)):
        diferenca = a[i] - b[i]

        if diferenca < 0:
            diferenca = -diferenca

        if diferenca > maior:
            maior = diferenca

    return maior
# K-NN: k-Nearest Neighbors ============================================================================================
class KNN:
    def __init__(self, k, tipo_distancia):
        self.k = k
        self.tipo_distancia = tipo_distancia

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def calcular_distancia(self, a, b):
        if self.tipo_distancia == "euclidiana":
            return distancia_euclidiana(a, b)
        elif self.tipo_distancia == "manhattan":
            return distancia_manhattan(a, b)
        elif self.tipo_distancia == "minkowski":
            return distancia_minkowski(a, b, 3)
        elif self.tipo_distancia == "chebyshev":
            return distancia_chebyshev(a, b)
        else:
            return distancia_euclidiana(a, b)

    def prever_um(self, x):

        distancias = []

        for i in range(len(self.X_train)):
            d = self.calcular_distancia(x, self.X_train[i])
            classe = self.y_train[i]
            distancias.append((d, classe))


        distancias.sort()


        vizinhos = []
        for i in range(self.k):
            vizinhos.append(distancias[i][1])


        contagem = {}
        for classe in vizinhos:
            if classe in contagem:
                contagem[classe] += 1
            else:
                contagem[classe] = 1

        classe_final = max(contagem, key=contagem.get)

        return classe_final

    def predict(self, X):
        resultados = []

        for x in X:
            resultados.append(self.prever_um(x))

        return resultados
# Testando no Iris =====================================================================================================
from sklearn import datasets
import random

iris = datasets.load_iris()

X = iris.data
y = iris.target

dados = list(zip(X, y))
random.shuffle(dados)

X, y = zip(*dados)

X = list(X)
y = list(y)

tamanho_treino = int(0.8 * len(X))

X_train = X[:tamanho_treino]
X_test = X[tamanho_treino:]

y_train = y[:tamanho_treino]
y_test = y[tamanho_treino:]

def acuracia(y_real, y_pred):
    acertos = 0

    for i in range(len(y_real)):
        if y_real[i] == y_pred[i]:
            acertos += 1

    return acertos / len(y_real)

distancias = ["euclidiana", "manhattan", "minkowski", "chebyshev"]

for d in distancias:
    modelo = KNN(k=3, tipo_distancia=d)
    modelo.fit(X_train, y_train)

    previsoes = modelo.predict(X_test)

    acc = acuracia(y_test, previsoes)

    print("Distância:", d)
    print("Acurácia:", acc)
    print()