import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from collections import Counter

# Carregar o dataset ajustado
df = pd.read_csv('classificacao_ajustado.csv')

# Remover a coluna 'id'
df = df.drop(columns='id')

# Separar atributos e rótulos
X = df.drop(columns='label')
y = df['label']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Matriz inversa da covariância (para Mahalanobis)
cov = np.cov(X_train, rowvar=False)
VI = np.linalg.inv(cov)

# Função para calcular distâncias
def calcular_distancia(x1, x2, tipo, VI=None):
    if tipo == 'euclidiana':
        return np.linalg.norm(x1 - x2)
    elif tipo == 'manhattan':
        return np.sum(np.abs(x1 - x2))
    elif tipo == 'chebyshev':
        return np.max(np.abs(x1 - x2))
    elif tipo == 'mahalanobis':
        return distance.mahalanobis(x1, x2, VI)
    else:
        raise ValueError("Tipo de distância inválido")

# Função KNN manual
def knn_manual(X_train, y_train, X_test, k, tipo, VI=None):
    y_pred = []
    for x_test in X_test:
        distancias = [(calcular_distancia(x_test, x_train, tipo, VI), y)
                      for x_train, y in zip(X_train, y_train)]
        distancias.sort(key=lambda x: x[0])
        k_vizinhos = [label for _, label in distancias[:k]]
        pred = Counter(k_vizinhos).most_common(1)[0][0]
        y_pred.append(pred)
    return y_pred

# Avaliação com diferentes distâncias
distancias = ['euclidiana', 'manhattan', 'chebyshev', 'mahalanobis']
k = 5
resultados = {}

for dist in distancias:
    print(f"\nDistância: {dist}")
    y_pred = knn_manual(X_train, y_train.values, X_test, k, dist, VI)
    acuracia = np.mean(y_pred == y_test.values)
    resultados[dist] = acuracia
    print(f"Acurácia: {acuracia:.4f}")

# Mostrar resumo final
print("\n Acurácias:")
for dist, acc in resultados.items():
    print(f"{dist.capitalize():<12}: {acc:.4f}")
