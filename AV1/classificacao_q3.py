import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

# Tratar valores ausentes
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Separar variável alvo e codificar variáveis categóricas
y = df['class'].map({'p': 0, 'e': 1})  # Exemplo de codificação binária
X = pd.get_dummies(df.drop('class', axis=1)).astype(float)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Função KNN manual usando numpy (distância Euclidiana)
def knn(X_train, y_train, X_test, k=5):
    y_pred = []
    for test_point in X_test:
        dists = np.linalg.norm(X_train - test_point, axis=1)
        k_indices = np.argsort(dists)[:k]
        k_labels = y_train.iloc[k_indices]
        y_pred.append(k_labels.mode()[0])
    return y_pred

# Normalizações
normalizacoes = {
    "Sem Normalização": (lambda x: x),
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler()
}

for nome, normalizador in normalizacoes.items():
    if nome == "Sem Normalização":
        X_train_n = X_train.values
        X_test_n = X_test.values
    else:
        X_train_n = normalizador.fit_transform(X_train)
        X_test_n = normalizador.transform(X_test)

    y_pred = knn(X_train_n, y_train, X_test_n, k=5)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia com {nome}: {acc:.4f}")
