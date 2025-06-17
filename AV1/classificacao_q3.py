# Recarregar os dados
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from scipy.spatial import distance
from collections import Counter

# Dataset
df = pd.read_csv('C:/Users/xulia/IA-CD/IA-CD/AV1/classificacao_ajustado.csv')
df['Medicamento'] = df['Medicamento'].astype(int)
X_raw = pd.get_dummies(df.drop(columns=['Medicamento']), drop_first=True).astype(float)
y = df['Medicamento']

# Função de KNN manual
def knn_predict(X_train, y_train, X_test_instance, k, metric='mahalanobis', VI=None):
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

# Avaliação
def evaluate_knn(X_train, y_train, X_test, y_test, k, metric='mahalanobis'):
    VI = None
    if metric == 'mahalanobis':
        VI = np.linalg.inv(np.cov(X_train.T))
    predictions = []
    for i in range(len(X_test)):
        pred = knn_predict(X_train, y_train, X_test.iloc[i], k, metric, VI)
        predictions.append(pred)
    return np.mean(np.array(predictions) == y_test.to_numpy())

# Normalizações
normalizations = {
    'sem_normalizacao': FunctionTransformer(func=None),
    'logaritmica': FunctionTransformer(np.log1p, validate=True),
    'minmax': MinMaxScaler(),
    'standard': StandardScaler()
}

# Resultados para cada normalização com k=5
results_q3 = {}
for name, transformer in normalizations.items():
    X_transformed = transformer.fit_transform(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        pd.DataFrame(X_transformed, columns=X_raw.columns),
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )
    acc = evaluate_knn(X_train, y_train, X_test, y_test, k=5, metric='mahalanobis')
    results_q3[name] = acc

# Mostrar resultado
results_q3_df = pd.DataFrame.from_dict(results_q3, orient='index', columns=['Acurácia'])
results_q3_df.index.name = "Normalização"
print("Acurácia por Normalização - KNN Manual")
print(results_q3_df)
