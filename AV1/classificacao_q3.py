import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Leitura do dataset
df = pd.read_csv('dataset/classificacao_ajustado.csv')

# Separação de dados
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Split original
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Distância Euclidiana (melhor do Q2)
def dist_euclidiana(v1, v2):
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(len(v1) - 1)))

# Função KNN manual
def knn(treinamento, nova_amostra, K, distancia_func):
    dists = {}
    for i, amostra in enumerate(treinamento):
        d = distancia_func(amostra, nova_amostra)
        dists[i] = d
    k_vizinhos = sorted(dists, key=dists.get)[:K]
    votos = [treinamento[i][-1] for i in k_vizinhos]
    votos = [float(v.item()) if isinstance(v, np.ndarray) else float(v) for v in votos]
    return max(set(votos), key=votos.count)

# Avaliação
def avaliar(treinamento, teste, distancia_func, K):
    acertos = 0
    for amostra in teste:
        classe_predita = knn(treinamento, amostra, K, distancia_func)
        if classe_predita == amostra[-1]:
            acertos += 1
    return 100 * acertos / len(teste)

# Transforma dataset (normalização + formatação para KNN)
def preparar_dataset(X_train, X_test, y_train, y_test):
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values
    y_test_np = y_test.values
    treinamento = [list(X_train_np[i]) + [y_train_np[i]] for i in range(len(X_train_np))]
    teste = [list(X_test_np[i]) + [y_test_np[i]] for i in range(len(X_test_np))]
    return treinamento, teste

# Avalia com normalização especificada
def avaliar_normalizacao(nome, X_train_norm, X_test_norm):
    treinamento, teste = preparar_dataset(X_train_norm, X_test_norm, y_train, y_test)
    acc = avaliar(treinamento, teste, dist_euclidiana, K=3)
    print(f"Acurácia com {nome}: {acc:.2f}%")

# ➤ Sem normalização
treinamento_orig, teste_orig = preparar_dataset(X_train_orig, X_test_orig, y_train, y_test)
acc_orig = avaliar(treinamento_orig, teste_orig, dist_euclidiana, K=3)
print(f"Acurácia sem normalização: {acc_orig:.2f}%")

# ➤ Logarítmica (usa log1p para lidar com zeros)
X_train_log = np.log1p(X_train_orig)
X_test_log = np.log1p(X_test_orig)
avaliar_normalizacao("logarítmica", X_train_log, X_test_log)

# ➤ MinMax
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train_orig)
X_test_minmax = scaler_minmax.transform(X_test_orig)
avaliar_normalizacao("MinMax", pd.DataFrame(X_train_minmax), pd.DataFrame(X_test_minmax))

# ➤ StandardScaler
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train_orig)
X_test_std = scaler_std.transform(X_test_orig)
avaliar_normalizacao("StandardScaler", pd.DataFrame(X_train_std), pd.DataFrame(X_test_std))

# ➤ Plot acurácia vs K
def plot_acuracia_vs_K(X_train, X_test, y_train, y_test, max_k=20):
    treinamento, teste = preparar_dataset(X_train, X_test, y_train, y_test)
    treino_accuracies = []
    teste_accuracies = []

    for k in range(1, max_k + 1):
        acc_treino = avaliar(treinamento, treinamento, dist_euclidiana, K=k)
        acc_teste = avaliar(treinamento, teste, dist_euclidiana, K=k)
        treino_accuracies.append(acc_treino)
        teste_accuracies.append(acc_teste)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), treino_accuracies, label="Treino", marker='o')
    plt.plot(range(1, max_k + 1), teste_accuracies, label="Teste", marker='s')
    plt.title("Acurácia vs. K (KNN com distância Euclidiana)")
    plt.xlabel("K")
    plt.ylabel("Acurácia (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, max_k + 1))
    plt.tight_layout()
    plt.show()

# Usamos a versão StandardScaler para análise de K (é comum ser a mais estável)
plot_acuracia_vs_K(pd.DataFrame(X_train_std), pd.DataFrame(X_test_std), y_train, y_test)
