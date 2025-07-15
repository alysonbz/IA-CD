import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Função de acurácia (caso ainda não tenha sido definida)
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Converter todos os conjuntos de treino e teste para numpy arrays
X_train_log = np.array(X_train_log)
X_test_log = np.array(X_test_log)
X_train_minmax = np.array(X_train_minmax)
X_test_minmax = np.array(X_test_minmax)
X_train_standard = np.array(X_train_standard)
X_test_standard = np.array(X_test_standard)
X_train_raw = np.array(X_train_raw)
X_test_raw = np.array(X_test_raw)

# Avaliar KNN com Chebyshev após conversão para arrays
results_normalizations = {}

for name, (X_tr, y_tr, X_te, y_te) in {
    "log": (X_train_log, y_train_log, X_test_log, y_test_log),
    "minmax": (X_train_minmax, y_train_minmax, X_test_minmax, y_test_minmax),
    "standard": (X_train_standard, y_train_standard, X_test_standard, y_test_standard),
    "sem_normalizacao": (X_train_raw, y_train_raw, X_test_raw, y_test_raw),
}.items():
    preds = knn_predict(X_tr, y_tr, X_te, k=5, metric='chebyshev')
    acc = accuracy(y_te, preds)
    results_normalizations[name] = acc

print("Resultados com diferentes normalizações:")
print(results_normalizations)

# Valores de k a testar
k_values = list(range(1, 31))

# Listas para armazenar acurácia em treino e teste
acc_train = []
acc_test = []

# Avaliar KNN para k de 1 a 30 usando Scikit-Learn
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='chebyshev')
    knn.fit(X_train_standard, y_train_standard)

    acc_train.append(knn.score(X_train_standard, y_train_standard))
    acc_test.append(knn.score(X_test_standard, y_test_standard))

# Plotar gráfico
plt.figure(figsize=(10, 5))
plt.plot(k_values, acc_train, label="Treino", marker='o', color='blue')
plt.plot(k_values, acc_test, label="Teste", marker='s', color='green')
plt.xlabel("Valor de k")
plt.ylabel("Acurácia")
plt.title("Acurácia vs k (StandardScaler + Chebyshev)")
plt.legend()
plt.grid(True)
plt.xticks(k_values)
plt.tight_layout()
plt.show()

# Encontrar o k com maior acurácia no teste
max_acc = max(acc_test)
best_k = k_values[acc_test.index(max_acc)]

print(f"Melhor k: {best_k} com acurácia: {max_acc:.4f}")