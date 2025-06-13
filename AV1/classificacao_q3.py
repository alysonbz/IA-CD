import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial import distance
from collections import Counter
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv('classificacao_ajustado.csv')
df = df.drop(columns='id')  # remover identificador
X = df.drop(columns='label')
y = df['label']

# Função para aplicar KNN manual
def calcular_distancia(x1, x2, tipo, VI=None):
    if tipo == 'chebyshev':
        return np.max(np.abs(x1 - x2))
    elif tipo == 'mahalanobis':
        return distance.mahalanobis(x1, x2, VI)
    elif tipo == 'euclidiana':
        return np.linalg.norm(x1 - x2)
    elif tipo == 'manhattan':
        return np.sum(np.abs(x1 - x2))

def knn_manual(X_train, y_train, X_test, k, tipo='chebyshev', VI=None):
    y_pred = []
    for x_test in X_test:
        distancias = [(calcular_distancia(x_test, x_train, tipo, VI), y)
                      for x_train, y in zip(X_train, y_train)]
        distancias.sort(key=lambda x: x[0])
        k_vizinhos = [label for _, label in distancias[:k]]
        pred = Counter(k_vizinhos).most_common(1)[0][0]
        y_pred.append(pred)
    return y_pred

# Função para testar uma normalização
def testar_knn_normalizacao(X_original, nome, normalizador=None, log=False):
    X_proc = X_original.copy()

    if log:
        # Só aplicar log às colunas numéricas
        for col in X_proc.select_dtypes(include=[np.number]).columns:
            X_proc[col] = X_proc[col].clip(lower=1e-5)  # evitar log(0) ou negativos
            X_proc[col] = np.log1p(X_proc[col])
    elif normalizador:
        X_proc = pd.DataFrame(normalizador().fit_transform(X_proc), columns=X_proc.columns)
    else:
        X_proc = X_proc.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.3, random_state=42
    )

    y_pred = knn_manual(np.array(X_train), y_train.values, np.array(X_test), k=5, tipo='chebyshev')
    acuracia = np.mean(y_pred == y_test.values)
    print(f"Acurácia com {nome}: {acuracia:.4f}")
    return nome, acuracia


# Comparar diferentes normalizações
print("\n--- Avaliação com diferentes normalizações ---")
resultados = []
resultados.append(testar_knn_normalizacao(X, 'Sem Normalização'))
resultados.append(testar_knn_normalizacao(X, 'Logarítmica', log=True))
resultados.append(testar_knn_normalizacao(X, 'Min-Max', normalizador=MinMaxScaler))
resultados.append(testar_knn_normalizacao(X, 'StandardScaler', normalizador=StandardScaler))

# Plot de acurácias
nomes, accs = zip(*resultados)
plt.figure(figsize=(8,5))
plt.bar(nomes, accs, color='skyblue')
plt.ylabel("Acurácia")
plt.title("Acurácia com diferentes normalizações (KNN, Chebyshev, k=5)")
plt.ylim(0, 1)
plt.show()

# Análise do valor de K
#  Usar a melhor normalização (StandardScaler)

X_norm = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.3, random_state=42
)

# Avaliar vários valores de k
ks = list(range(1, 21))
acc_treino = []
acc_teste = []

for k in ks:
    y_pred_train = knn_manual(X_train, y_train.values, X_train, k, tipo='chebyshev')
    y_pred_test = knn_manual(X_train, y_train.values, X_test, k, tipo='chebyshev')
    acc_treino.append(np.mean(y_pred_train == y_train.values))
    acc_teste.append(np.mean(y_pred_test == y_test.values))

# Plotar gráfico de acurácia vs k
plt.figure(figsize=(10, 6))
plt.plot(ks, acc_treino, marker='o', label='Treino')
plt.plot(ks, acc_teste, marker='s', label='Teste')
plt.xlabel("Valor de k")
plt.ylabel("Acurácia")
plt.title("Acurácia de Treino e Teste vs k (KNN com Chebyshev)")
plt.legend()
plt.grid(True)
plt.xticks(ks)
plt.show()