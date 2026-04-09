from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

wine = load_wine_dataset()
scaler = MinMaxScaler()

X = wine.drop(columns=['Quality'])
X = np.log(X)
y = wine['Quality'].values

# Divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

# Mostre quantos elementos do dataset estão faltando na coluna Quality
print("\nQuestão.")
print(wine['Quality'].isnull().sum())

# Aplique a função fit do knn
knn.fit(X_train, y_train)

# Mostre o acerto do algoritmo
print("\nQuestão.")
print(knn.score(X_test, y_test))