import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv("classificacao_ajustado.csv")
X = df.drop("class", axis=1)
y = df["class"]

# Aplicar normalizações
X_log = X.apply(lambda x: np.log1p(x))
X_minmax = MinMaxScaler().fit_transform(X)
X_std = StandardScaler().fit_transform(X)

datasets = {
    "Original": X.values,
    "Log": X_log.values,
    "MinMax": X_minmax,
    "Standard": X_std
}

# Função KNN
def knn(X_train, y_train, X_test, k):
    pred = []
    for test in X_test:
        dist = np.linalg.norm(X_train - test, axis=1)
        idx = np.argsort(dist)[:k]
        voto = Counter(y_train[idx]).most_common(1)[0][0]
        pred.append(voto)
    return pred

# Avaliar normalizações
for nome, dados in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(dados, y, test_size=0.3, random_state=42)
    y_pred = knn(X_train, y_train.values, X_test, 5)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia ({nome}): {acc:.4f}")

# Avaliação para diferentes valores de k
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)
train_acc = []
test_acc = []
ks = range(1, 21)

for k in ks:
    train_pred = knn(X_train, y_train.values, X_train, k)
    test_pred = knn(X_train, y_train.values, X_test, k)
    train_acc.append(accuracy_score(y_train, train_pred))
    test_acc.append(accuracy_score(y_test, test_pred))

# Plotar gráfico
plt.plot(ks, train_acc, label="Treino")
plt.plot(ks, test_acc, label="Teste")
plt.xlabel("k")
plt.ylabel("Acurácia")
plt.title("Acurácia x K")
plt.legend()
plt.savefig("knn_k_vs_acc.png")
