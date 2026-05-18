import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# CARREGAR DATASET
df = pd.read_csv('dataset_tratado.csv')

# SEPARAÇÃO DE X E Y
X = df.drop('Revenue', axis=1)
y = df['Revenue']


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# NORMALIZAÇÕES

# MIN-MAX
minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train)
X_test_minmax = minmax.transform(X_test)

# Z-SCORE
zscore = StandardScaler()
X_train_zscore = zscore.fit_transform(X_train)
X_test_zscore = zscore.transform(X_test)

# LOG
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

# LISTAS
acuracia_minmax = []
acuracia_zscore = []
acuracia_log = []

# TESTE EM DIFERENTES VALORES DE K
for k in range(1, 21):

    # MIN-MAX
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_minmax, y_train)
    pred = knn.predict(X_test_minmax)
    acuracia_minmax.append(accuracy_score(y_test, pred))

    # Z-SCORE
    knn.fit(X_train_zscore, y_train)
    pred = knn.predict(X_test_zscore)
    acuracia_zscore.append(accuracy_score(y_test, pred))

    # LOG
    knn.fit(X_train_log, y_train)
    pred = knn.predict(X_test_log)
    acuracia_log.append( accuracy_score(y_test, pred))


# MOSTRAR MELHORES RESULTADOS
print('Melhor Min-Max:', max(acuracia_minmax))
print('Melhor Z-score:', max(acuracia_zscore))
print('Melhor Log:', max(acuracia_log))

# Identificar o Melhor k

# Z-Score
melhor_k = acuracia_zscore.index(max(acuracia_zscore)) + 1
print("Melhor k Z-Score:", melhor_k)

# Min_Max
melhor_k_minmax = acuracia_minmax.index(max(acuracia_minmax)) + 1
print("Melhor k Min-Max:", melhor_k_minmax)

# LOG
melhor_k_log = acuracia_log.index(max(acuracia_log)) + 1
print("Melhor k Log:", melhor_k_log)


# MELHOR MODELO
knn_final = KNeighborsClassifier(n_neighbors=18)
knn_final.fit(X_train_log, y_train)
y_pred = knn_final.predict(X_test_log)


# MATRIZ DE CONFUSÃO
print('MATRIZ DE CONFUSÃO')
print(confusion_matrix(y_test, y_pred))

# CLASSIFICATION REPORT
print('CLASSIFICATION REPORT')
print(classification_report(y_test, y_pred))

# GRÁFICO
plt.plot(range(1, 21), acuracia_minmax, label='Min-Max')
plt.plot(range(1, 21), acuracia_zscore, label='Z-score')
plt.plot(range(1, 21), acuracia_log, label='Log')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia')
plt.title('KNN - Comparação das Normalizações')
plt.legend()
plt.grid(True)
#plt.savefig('grafico_knn.png')
plt.show()