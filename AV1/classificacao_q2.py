import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, chebyshev, mahalanobis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Carregar os dados
df = pd.read_csv('classificacao_ajustado.csv')

# Separar variáveis
X = df.drop(columns=['class_e'])
y = df['class_e']  # classe 'e' é venenoso

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preparar arrays
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Calcular matriz inversa da covariância para Mahalanobis
VI = np.linalg.inv(np.cov(X_train_np.T) + np.eye(X_train_np.shape[1]) * 1e-6)

# Função KNN manual
def knn(X_train, y_train, X_test, k, metric, VI=None):
    y_pred = []
    for test_point in X_test:
        if metric == mahalanobis:
            dists = [metric(test_point, x, VI) for x in X_train]
        else:
            dists = [metric(test_point, x) for x in X_train]
        k_indices = np.argsort(dists)[:k]
        k_labels = y_train.iloc[k_indices]
        majority = Counter(k_labels).most_common(1)[0][0]
        y_pred.append(majority)
    return np.array(y_pred)

# Avaliação com diferentes métricas
metricas = {
    'Euclidiana': euclidean,
    'Manhattan': cityblock,
    'Chebyshev': chebyshev,
    'Mahalanobis': lambda u, v: mahalanobis(u, v, VI)
}

for nome, metrica in metricas.items():
    y_pred = knn(X_train_np, y_train, X_test_np, k=5, metric=metrica)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia com {nome}: {acc:.4f}")
