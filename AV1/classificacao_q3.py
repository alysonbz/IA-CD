import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# ============================
# 1. Carregar e preparar os dados
# ============================

df = pd.read_csv("IA-CD/AV1/bancos/flavors_of_cacao.csv")

# Corrigir os nomes das colunas
df.columns = [
    "Company", "Specific_Bean_Origin", "REF", "Review_Date", "Cocoa_Percent",
    "Company_Location", "Rating", "Bean_Type", "Broad_Bean_Origin"
]

# Limpar porcentagem e converter para número
df["Cocoa_Percent"] = df["Cocoa_Percent"].str.replace("%", "").astype(float)

# Selecionar apenas colunas numéricas
X = df[["REF", "Review_Date", "Cocoa_Percent"]].values
y = df["Rating"].values

# ============================
# 2. Funções de distância
# ============================

def euclidean(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

def manhattan(v1, v2):
    return np.sum(np.abs(v1 - v2))

def minkowski(v1, v2, p=3):
    return np.sum(np.abs(v1 - v2) ** p) ** (1/p)

# ============================
# 3. Função KNN Manual
# ============================

def knn_predict(X_train, y_train, X_test, k, dist_func):
    predictions = []
    for test_point in X_test:
        distances = [dist_func(test_point, x) for x in X_train]
        idx = np.argsort(distances)[:k]
        nearest_labels = y_train[idx]
        pred = np.mean(nearest_labels)  # Regressão KNN
        predictions.append(pred)
    return np.array(predictions)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# ============================
# 4. Separar treino e teste
# ============================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar normalizações
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

# ============================
# 5. Testar diferentes distâncias e normalizações
# ============================

k = 3
results = {}

for name, X_tr, X_te in [
    ("Sem Normalização", X_train, X_test),
    ("Log", X_train_log, X_test_log),
    ("MinMax", X_train_minmax, X_test_minmax),
    ("Standard", X_train_std, X_test_std)
]:
    for dist_name, dist_func in [
        ("Euclidean", euclidean),
        ("Manhattan", manhattan),
        ("Minkowski", lambda v1, v2: minkowski(v1, v2, p=3))
    ]:
        y_pred = knn_predict(X_tr, y_train, X_te, k, dist_func)
        error = rmse(y_test, y_pred)
        results[(name, dist_name)] = error

# Mostrar os resultados
print("\n===== RMSE para cada combinação de normalização e distância =====")
for (norm, dist), error in results.items():
    print(f"{norm} + {dist} -> RMSE Teste: {error:.4f}")

# Melhor configuração
best_setting = min(results, key=results.get)
melhor_normalizacao, melhor_distancia = best_setting
print(f"\n✅ Melhor configuração: {melhor_normalizacao} + {melhor_distancia}")

# ============================
# 6. Análise de K (gráfico)
# ============================

# Escolher os dados normalizados
if melhor_normalizacao == "MinMax":
    X_train_best, X_test_best = X_train_minmax, X_test_minmax
elif melhor_normalizacao == "Standard":
    X_train_best, X_test_best = X_train_std, X_test_std
elif melhor_normalizacao == "Log":
    X_train_best, X_test_best = X_train_log, X_test_log
else:
    X_train_best, X_test_best = X_train, X_test

# Função de distância
if melhor_distancia == "Euclidean":
    dist_func_best = euclidean
elif melhor_distancia == "Manhattan":
    dist_func_best = manhattan
else:
    dist_func_best = lambda v1, v2: minkowski(v1, v2, p=3)

# Pré-calcular matrizes de distância para otimizar
train_dist_matrix = np.array([[dist_func_best(x1, x2) for x2 in X_train_best] for x1 in X_train_best])
test_dist_matrix = np.array([[dist_func_best(x1, x2) for x2 in X_train_best] for x1 in X_test_best])

def fast_knn(y_train, dist_matrix, k):
    preds = []
    for distances in dist_matrix:
        idx = np.argsort(distances)[:k]
        preds.append(np.mean(y_train[idx]))
    return np.array(preds)

train_rmse_list = []
test_rmse_list = []
k_values = range(1, 21)

for k in k_values:
    y_train_pred = fast_knn(y_train, train_dist_matrix, k)
    y_test_pred = fast_knn(y_train, test_dist_matrix, k)
    
    train_rmse_list.append(rmse(y_train, y_train_pred))
    test_rmse_list.append(rmse(y_test, y_test_pred))

# Plot
plt.figure(figsize=(8,5))
plt.plot(k_values, train_rmse_list, label='Train RMSE', marker='o')
plt.plot(k_values, test_rmse_list, label='Test RMSE', marker='s')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.title(f'KNN - RMSE vs k ({melhor_normalizacao} + {melhor_distancia})')
plt.legend()
plt.tight_layout()
plt.show()
