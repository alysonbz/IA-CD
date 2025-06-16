import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from scipy.spatial import distance

# 1. Carregar dados
df = pd.read_csv("classificacao_ajustado.csv")
X = df.drop("class", axis=1).values
y = df["class"].values

# 2. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Função KNN
def knn(X_train, y_train, X_test, k, dist_func):
    predictions = []
    for test_point in X_test:
        distances = [dist_func(test_point, x) for x in X_train]
        k_idx = np.argsort(distances)[:k]
        k_labels = y_train[k_idx]
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return predictions

# 4. Funções de distância
distancias = {
    "euclidiana": lambda a, b: np.linalg.norm(a - b),
    "manhattan": lambda a, b: distance.cityblock(a, b),
    "chebyshev": lambda a, b: distance.chebyshev(a, b),
    "mahalanobis": lambda a, b: distance.mahalanobis(a, b, np.linalg.inv(np.cov(X_train.T)))
}

# 5. Avaliação
for nome, func in distancias.items():
    y_pred = knn(X_train, y_train, X_test, k=5, dist_func=func)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia usando distância {nome}: {acc:.4f}")
