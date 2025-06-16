import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

df = pd.read_csv(r"C:\Users\jsslu\OneDrive\Área de Trabalho\UFC\Inteligencia Artificial\git\IA-CD\AV1\datasets\classificacao_ajustado.csv")

X = df.drop('label', axis=1).values
y = df['label'].values

# Divida o dataset em treino e teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#distancia euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

#distancia manhattan
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

#distancia chebyshev
def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))

#distancia mahalanobis
def mahalanobis_distance(x1, x2, cov_inv):
    diff = x1 - x2
    return np.sqrt(diff.T @ cov_inv @ diff)


#Implemente manualmente o KNN
def predict_classification(X_train, y_train, test_row, k, distance_func, cov_inv=None):
    distances = []
    for i, train_row in enumerate(X_train):
        if distance_func == mahalanobis_distance:
            dist = distance_func(test_row, train_row, cov_inv)
        else:
            dist = distance_func(test_row, train_row)
        distances.append((y_train[i], dist))


    distances.sort(key=lambda item: item[1])


    neighbors = [item[0] for item in distances[:k]]


    most_common = Counter(neighbors).most_common(1)
    return most_common[0][0]

def knn(X_train, y_train, X_test, k, distance_func):
    cov_inv = None
    if distance_func == mahalanobis_distance:
        # Calcula a matriz de covariância a partir dos dados de treino
        cov_matrix = np.cov(X_train, rowvar=False)
        # Calcula a inversa da matriz de covariância
        cov_inv = np.linalg.pinv(cov_matrix)

    predictions = []
    for test_row in X_test:
        prediction = predict_classification(X_train, y_train, test_row, k, distance_func, cov_inv)
        predictions.append(prediction)
    return np.array(predictions)

def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions


K = 6

results = {}

# Avaliação com Distância Euclidiana
print("Calculando com a distância Euclidiana...")
predictions_euclidean = knn(X_train, y_train, X_test, K, euclidean_distance)
accuracy_euclidean = calculate_accuracy(y_test, predictions_euclidean)
results['Euclidiana'] = accuracy_euclidean

# Avaliação com Distância Manhattan
print("Calculando com a distância de Manhattan...")
predictions_manhattan = knn(X_train, y_train, X_test, K, manhattan_distance)
accuracy_manhattan = calculate_accuracy(y_test, predictions_manhattan)
results['Manhattan'] = accuracy_manhattan

# Avaliação com Distância Chebyshev
print("Calculando com a distância de Chebyshev...")
predictions_chebyshev = knn(X_train, y_train, X_test, K, chebyshev_distance)
accuracy_chebyshev = calculate_accuracy(y_test, predictions_chebyshev)
results['Chebyshev'] = accuracy_chebyshev

# Avaliação com Distância Mahalanobis
print("Calculando com a distância de Mahalanobis...")
predictions_mahalanobis = knn(X_train, y_train, X_test, K, mahalanobis_distance)
accuracy_mahalanobis = calculate_accuracy(y_test, predictions_mahalanobis)
results['Mahalanobis'] = accuracy_mahalanobis


# Compare os resultados obtidos para os diferentes valores de distancia, considerando a métrica acurácia.

print("\nComparação dos Resultados (Acurácia)")
print(f"Valor de K (vizinhos): {K}\n")

for metric, accuracy in results.items():
    print(f"Distância {metric}: {accuracy:.4f} ({accuracy:.2%})")

print("\n--- Conclusão ---")
best_metric = max(results, key=results.get)
print(f"A melhor métrica de distância para este problema (com K={K}) foi a de {best_metric},")
print(f"com uma acurácia de {results[best_metric]:.2%}.")

