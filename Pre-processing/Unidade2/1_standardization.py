import numpy as np
from sklearn.preprocessing import MinMaxScaler #scaler.fit_transform(X)
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

wine = load_wine_dataset()
print(wine.describe())

scaler = MinMaxScaler()

X = wine.drop(['Quality'],axis=1)

X = np.log(X)

y = wine['Quality'].values

# divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()

# Aplique a função fit do knn
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

# mostre o acerto do algoritmo
print(knn.score(X_test, y_test))

print(pred)