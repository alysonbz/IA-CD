import pandas as pd

from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wine = load_wine_dataset()

X = wine.drop(['Quality'],axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = wine['Quality'].values

# divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier()

# Aplique a função fit do knn
knn.fit(X_train, y_train)

# mostre o acerto do algoritmo
print(knn.score(X,y))
