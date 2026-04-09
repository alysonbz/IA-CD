# Import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

wine = load_wine_dataset()

# Inicialize o scale
scaler = StandardScaler()

# Exclua do dataset a coluna
X = wine.drop(columns=['Quality'])

# Normalize o dataset com scaler
X_norm = scaler.fit_transform(X)

# Obtenha as labels da coluna Quality
y = wine['Quality'].values

# Print a variância de X
print("\nQuestão.")
print('Variância', X.var())

# Print a variância do dataset X_norm
print("\nQuestão.")
print('Variância do dataset normalizado', X_norm.var())

# Divida o dataset em treino e teste com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, stratify=y, random_state=42)

# Inicialize o algoritmo KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Aplique a função fit do KNN
knn.fit(X_train, y_train)

# Verifique o acerto do classificador
print("\nQuestão.")
print('score', knn.score(X_test, y_test))