# 1. Importar StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

wine = load_wine_dataset()

scaler = StandardScaler()

X = wine.drop(['Quality'], axis=1)

X_norm = scaler.fit_transform(X)

y = wine['Quality'].values

print('Variancia X:', np.var(X))

print('Variancia X_norm', np.var(X_norm))

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, stratify=y, random_state=42
)

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

print('score', knn.score(X_test, y_test))
