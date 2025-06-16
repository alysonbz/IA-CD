#Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Importando o csv com o pandas
df = pd.read_csv('C:/Users/xulia/IA-CD/IA-CD/AV1/classificacao_ajustado.csv')
print("________________________________________________________")


# Garantir que 'Medicamento' seja inteiro (classe alvo)
df['Medicamento'] = df['Medicamento'].astype(int)

# Transformar variáveis categóricas (PA, Sexo, Colesterol) em numéricas
df_encoded = pd.get_dummies(df.drop('Medicamento', axis=1), drop_first=True)



# Separar X e y
X = df_encoded.astype(float)
y = df['Medicamento'].astype(int)

# Divisão treino/teste estratificada
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Função de KNN manual com várias distâncias
from scipy.spatial import distance
from collections import Counter


def knn_predict(X_train, y_train, X_test_instance, k, metric='euclidean', VI=None):
    distancias = []
    for i in range(len(X_train)):
        if metric == 'euclidean':
            dist = distance.euclidean(X_train.iloc[i], X_test_instance)
        elif metric == 'manhattan':
            dist = distance.cityblock(X_train.iloc[i], X_test_instance)
        elif metric == 'chebyshev':
            dist = distance.chebyshev(X_train.iloc[i], X_test_instance)
        elif metric == 'mahalanobis':
            dist = distance.mahalanobis(X_train.iloc[i], X_test_instance, VI)
        else:
            raise ValueError("Métrica inválida.")
        distancias.append((dist, y_train.iloc[i]))
    distancias.sort(key=lambda x: x[0])
    k_vizinhos = [label for _, label in distancias[:k]]
    return Counter(k_vizinhos).most_common(1)[0][0]


# Avaliação da acurácia para cada métrica
def evaluate_knn(X_train, y_train, X_test, y_test, k, metric):
    VI = None
    if metric == 'mahalanobis':
        VI = np.linalg.inv(np.cov(X_train.T))

    predictions = []
    for i in range(len(X_test)):
        pred = knn_predict(X_train, y_train, X_test.iloc[i], k, metric, VI)
        predictions.append(pred)

    predictions = np.array(predictions)
    return np.mean(predictions == y_test.to_numpy())


# Avaliar com todas as métricas
metrics = ['euclidean', 'manhattan', 'chebyshev', 'mahalanobis']
results = {m: evaluate_knn(X_train, y_train, X_test, y_test, k=5, metric=m) for m in metrics}

# Mostrar resultado
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Acurácia'])
results_df.index.name = "Métrica"
print("Resultados de Acurácia - KNN Manual")
print(results_df)

