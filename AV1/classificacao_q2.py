import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Carregar o dataset
df = pd.read_csv("classificacao_ajustado.csv")

# Reduzir para 1000 amostras aleatórias
df = df.sample(n=1000, random_state=42)

# Selecionar colunas numéricas úteis
X = df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']].copy()
y = df['class'].copy()

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# Funções de distância
def dist_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def dist_manhattan(a, b):
    return np.sum(np.abs(a - b))

def dist_chebyshev(a, b):
    return np.max(np.abs(a - b))

def dist_minkowski(a, b, p=3):
    return np.sum(np.abs(a - b) ** p) ** (1/p)

# Implementação do KNN manual
def knn_predict(X_train, y_train, x_test, k, distance_func):
    distances = []
    for i in range(len(X_train)):
        dist = distance_func(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    votes = [label for _, label in neighbors]
    most_common = Counter(votes).most_common(1)
    return most_common[0][0]

# Avaliar desempenho com cada distância
def evaluate_knn(distance_func, k=5):
    predictions = []
    for x in X_test:
        pred = knn_predict(X_train, y_train, x, k, distance_func)
        predictions.append(pred)
    accuracy = np.mean(predictions == y_test)
    return accuracy

# Testar todas as distâncias
results = {}
results['Euclidiana'] = evaluate_knn(dist_euclidiana)
results['Manhattan'] = evaluate_knn(dist_manhattan)
results['Chebyshev'] = evaluate_knn(dist_chebyshev)
results['Minkowski_p3'] = evaluate_knn(lambda a, b: dist_minkowski(a, b, p=3))

# Mostrar resultados
for dist_name, acc in results.items():
    print(f"Acurácia com distância {dist_name}: {acc:.4f}")

melhor = max(results, key=results.get)
print(f"\nMelhor distância foi: {melhor} com acurácia de {results[melhor]:.4f}")
