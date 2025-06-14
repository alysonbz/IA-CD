import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter



# Carregar o dataset
df = pd.read_csv("IA-CD/AV1/bancos/flavors_of_cacao.csv")
df.columns = [
    'company',
    'bean_origin',
    'ref',
    'review_date',
    'cocoa_percent',
    'company_location',
    'rating',
    'bean_type',
    'broad_bean_origin'
]
#  Pré-processamento
# Exemplo de tratamento básico - adapte conforme necessário
df.columns = [col.strip() for col in df.columns]
df = df.dropna()

# Transformando variáveis categóricas
df['company'] = pd.factorize(df['company'])[0]
df['company_location'] = pd.factorize(df['company_location'])[0]
df['bean_type'] = pd.factorize(df['bean_type'])[0]
df['broad_bean_origin'] = pd.factorize(df['broad_bean_origin'])[0]

# Definindo features e target
X = df[['company', 'company_location', 'cocoa_percent', 'bean_type', 'broad_bean_origin']].copy()
X['cocoa_percent'] = X['cocoa_percent'].apply(lambda x: float(str(x).strip('%')))  # limpar o percentual
y = (df['rating'] >= df['rating'].mean()).astype(int)  # Exemplo: classificar como "bom" (1) ou "ruim" (0)

#  Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

#  Funções de distância
def dist_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def dist_manhattan(a, b):
    return np.sum(np.abs(a - b))

def dist_chebyshev(a, b):
    return np.max(np.abs(a - b))

def dist_minkowski(a, b, p=3):
    return np.sum(np.abs(a - b) ** p) ** (1/p)

#  Implementação do KNN manual
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

#  Avaliar desempenho com cada distância
def evaluate_knn(distance_func, k=5):
    predictions = []
    for x in X_test:
        pred = knn_predict(X_train, y_train, x, k, distance_func)
        predictions.append(pred)
    accuracy = np.mean(predictions == y_test)
    return accuracy

#  Avaliar todas as distâncias
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
