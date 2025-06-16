import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

# Carregamento do dataset
df = pd.read_csv("./dataset/classificacao_ajustado.csv")

X = df.drop('label', axis=1).values
y = df['label'].values

def mahalanobis_distance(x1, x2, cov_inv):
    diff = x1 - x2
    return np.sqrt(diff.T @ cov_inv @ diff)

# KNN manual
def predict_classification(X_train, y_train, test_row, k, cov_inv):
    distances = []
    for i, train_row in enumerate(X_train):
        dist = mahalanobis_distance(test_row, train_row, cov_inv)
        distances.append((y_train[i], dist))
    distances.sort(key=lambda item: item[1])
    neighbors = [item[0] for item in distances[:k]]
    most_common = Counter(neighbors).most_common(1)
    return most_common[0][0]

def knn(X_train, y_train, X_test, k):
    cov_matrix = np.cov(X_train, rowvar=False)
    cov_inv = np.linalg.pinv(cov_matrix)
    predictions = []
    for test_row in X_test:
        prediction = predict_classification(X_train, y_train, test_row, k, cov_inv)
        predictions.append(prediction)
    return np.array(predictions)

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Função auxiliar para avaliação
def avaliar_knn(X, y, normalizador, nome):
    if normalizador is not None:
        X = normalizador.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = knn(X_train, y_train, X_test, k=5)
    acc = calculate_accuracy(y_test, y_pred)
    print(f"Acurácia com {nome}: {acc:.4f}")
    return acc

# Avaliação com e sem normalização
print("--- Avaliação de Acurácia com diferentes normalizações ---")
acc_orig = avaliar_knn(X, y, None, "sem normalização")
acc_log = avaliar_knn(np.log1p(X), y, None, "logarítmica")
acc_minmax = avaliar_knn(X, y, MinMaxScaler(), "MinMaxScaler")
acc_std = avaliar_knn(X, y, StandardScaler(), "StandardScaler")

# Gráfico de comparação
labels = ['Sem normalização', 'Logarítmica', 'MinMax', 'Standard']
accs = [acc_orig, acc_log, acc_minmax, acc_std]

plt.figure(figsize=(8, 5))
plt.bar(labels, accs, color='blue')
plt.title("Acurácia do KNN (Mahalanobis) com diferentes normalizações")
plt.ylabel("Acurácia")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.xticks(rotation=15)
plt.show()

#Faça também uma análise manual para descobrir o melhor valor de K para o KNN. Plote o gráfico de acurácia de treino e teste versus `k`.
# Análise do melhor K com dados padronizados (StandardScaler)

# --- Reaplica StandardScaler para garantir dados corretamente normalizados ---
X_std = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

k_values = list(range(1, 10))
train_accuracies = []
test_accuracies = []

cov_inv = np.linalg.pinv(np.cov(X_train, rowvar=False))

for k in k_values:
    print(f"Calculando K = {k}")
    y_train_pred = [predict_classification(X_train, y_train, row, k, cov_inv) for row in X_train]
    y_test_pred = [predict_classification(X_train, y_train, row, k, cov_inv) for row in X_test]
    train_accuracies.append(calculate_accuracy(y_train, y_train_pred))
    test_accuracies.append(calculate_accuracy(y_test, y_test_pred))

best_k_index = np.argmax(test_accuracies)
best_k = k_values[best_k_index]
best_acc = test_accuracies[best_k_index]

# Gráfico da acurácia por K
plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracies, label='Acurácia Treino', marker='o')
plt.plot(k_values, test_accuracies, label='Acurácia Teste', marker='s')
plt.axvline(best_k, color='red', linestyle='--', label=f'Melhor K = {best_k}')
plt.title('Acurácia vs. Valor de K (com StandardScaler - Mahalanobis)')
plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Resultado final
print(f"\nMelhor valor de K: {best_k}")
print(f"Acurácia no teste com K={best_k}: {best_acc:.4f} ({best_acc:.2%})")