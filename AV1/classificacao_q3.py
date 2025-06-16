# 1. Aplique as normalizações: logarítmica, minMax, StandardScaler.

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# Logaritmica

x_train_log = np.log(x_train)
print("Normalização Logarítmica: ", x_train_log)

x_test_log = np.log(x_test)
print("Normalização Logarítmica: ", x_test_log)

##

#MinMaX

scaler = MinMaxScaler()
x_train_minmax = scaler.fit_transform(x_train)
x_test_minmax = scaler.transform(x_test)

print("Normalização MinMaxScaler: ", x_train_minmax)
print(x_train_minmax)
print("Normalização MinMaxScaler: ", x_test_minmax)
print(x_test_minmax) 

##

# StandardScaler
scaler_std = StandardScaler()

x_train_std = scaler_std.fit_transform(x_train)
print("Normalização StandardScaler:", x_train_std)

x_test_std = scaler_std.transform(x_test)
print("Normalização StandardScaler:", x_test_std)

##

# 2. Reaplique o KNN manual com a melhor distancia

def avaliar_knn(X_train, y_train, X_test, y_test, k=5, metric="euclidean"):
    y_pred = knn_predict(X_train, y_train, X_test, k=k, metric=metric)
    return accuracy_score(y_test, y_pred)

melhor_dist = "euclidean"

def to_np_if_pandas(x):
    import pandas as pd
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    return x

print("Sem normalização:", 
      avaliar_knn(to_np_if_pandas(x_train), to_np_if_pandas(y_train), 
                  to_np_if_pandas(x_test), to_np_if_pandas(y_test), 
                  k=5, metric=melhor_dist))

print("Logarítmica:", 
      avaliar_knn(to_np_if_pandas(x_train_log), to_np_if_pandas(y_train), 
                  to_np_if_pandas(x_test_log), to_np_if_pandas(y_test), 
                  k=5, metric=melhor_dist))

print("MinMaxScaler:", 
      avaliar_knn(x_train_minmax, to_np_if_pandas(y_train), 
                  x_test_minmax, to_np_if_pandas(y_test), 
                  k=5, metric=melhor_dist))

print("StandardScaler:", 
      avaliar_knn(x_train_std, to_np_if_pandas(y_train), 
                  x_test_std, to_np_if_pandas(y_test), 
                  k=5, metric=melhor_dist))

##

# 3. Compare os desempenhos com e sem normalização.

# Sem normalização: 0.6623
# Log: 0.6948
# MinMax = 0.6818
# Stand = 0.6948

# Sem normalização o resultado teve pior rendimento. Log, por sua vez obteve uma melhoria no rendimento dos dados. MinMax melhorou relativamente o desempenho dos dados, isso define que normalizar os dados melhora a performance do KNN. Por último Stand que teve a melhor acurácia, mostrando que os dados melhoram mais para uma distribuição padronizada.

##

# 4. Faça também uma análise manual para descobrir o melhor valor de K para o KNN. Plote o gráfico de acurácia de treino e teste versus k.

def to_numpy_if_pandas(x):
    import pandas as pd
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    return x

y_train_np = to_numpy_if_pandas(y_train)
y_test_np = to_numpy_if_pandas(y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(to_numpy_if_pandas(x_train))
X_test_scaled = scaler.transform(to_numpy_if_pandas(x_test))


# Testar valores de k de 1 até 20

k_values = range(1, 21)
acc_train = []
acc_test = []

for k in k_values:
    y_train_pred = knn_predict(X_train_scaled, y_train_np, X_train_scaled, k=k, metric="euclidean")
    acc_train.append(accuracy_score(y_train_np, y_train_pred))

    y_test_pred = knn_predict(X_train_scaled, y_train_np, X_test_scaled, k=k, metric="euclidean")
    acc_test.append(accuracy_score(y_test_np, y_test_pred))


# Gráfico

plt.figure(figsize=(10, 6))
plt.plot(k_values, acc_train, label="Acurácia Treino", marker='o')
plt.plot(k_values, acc_test, label="Acurácia Teste", marker='x')
plt.xlabel("Valor de K")
plt.ylabel("Acurácia")
plt.title("Acurácia vs Valor de K (KNN Manual)")
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

