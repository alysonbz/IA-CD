import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# 1. Carregar e reduzir o dataset
df = pd.read_csv("classificacao_ajustado.csv")
df = df.sample(n=1000, random_state=42)

# 2. Selecionar colunas numéricas úteis
X = df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']].copy()
y = df['class'].copy()

# 3. Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# 4. Distância Euclidiana (melhor da Q2)
def dist_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# 5. KNN manual
def knn_predict(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        dist = dist_euclidiana(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    votes = [label for _, label in neighbors]
    return Counter(votes).most_common(1)[0][0]

def evaluate_knn(X_train, y_train, X_test, y_test, k):
    pred_test = [knn_predict(X_train, y_train, x, k) for x in X_test]
    pred_train = [knn_predict(X_train, y_train, x, k) for x in X_train]
    acc_test = np.mean(pred_test == y_test)
    acc_train = np.mean(pred_train == y_train)
    return acc_train, acc_test

# 6. Normalizações
X_log_train = np.log1p(X_train)
X_log_test = np.log1p(X_test)

minmax = MinMaxScaler()
X_minmax_train = minmax.fit_transform(X_train)
X_minmax_test = minmax.transform(X_test)

standard = StandardScaler()
X_std_train = standard.fit_transform(X_train)
X_std_test = standard.transform(X_test)

# 7. Avaliar k=5 com e sem normalização
acc_log = evaluate_knn(X_log_train, y_train, X_log_test, y_test, k=5)
acc_minmax = evaluate_knn(X_minmax_train, y_train, X_minmax_test, y_test, k=5)
acc_std = evaluate_knn(X_std_train, y_train, X_std_test, y_test, k=5)
acc_orig = evaluate_knn(X_train, y_train, X_test, y_test, k=5)

print("\nAcurácias com k=5:")
print(f"Sem normalização: Treino={acc_orig[0]:.4f} | Teste={acc_orig[1]:.4f}")
print(f"Logarítmica:       Treino={acc_log[0]:.4f} | Teste={acc_log[1]:.4f}")
print(f"MinMaxScaler:      Treino={acc_minmax[0]:.4f} | Teste={acc_minmax[1]:.4f}")
print(f"StandardScaler:    Treino={acc_std[0]:.4f} | Teste={acc_std[1]:.4f}")

# 8. Análise manual do melhor k (usando dados normalizados com StandardScaler)
k_values = range(1, 21)
acc_train_list = []
acc_test_list = []

for k in k_values:
    acc_train, acc_test = evaluate_knn(X_std_train, y_train, X_std_test, y_test, k)
    acc_train_list.append(acc_train)
    acc_test_list.append(acc_test)

# 9. Plotar gráfico
plt.plot(k_values, acc_train_list, label='Treino')
plt.plot(k_values, acc_test_list, label='Teste')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.title('Acurácia vs k (com StandardScaler)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
