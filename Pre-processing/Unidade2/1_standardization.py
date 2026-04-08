from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

wine = load_wine_dataset()

print(wine.describe())

#MinMaxScaler, tipo de normalização, ele deixa de variar (Ex.: 1 até 10 ou 15 até 25),
#fazendo com que o mínino e o máximo seja entre 0 e 1.
scaler = MinMaxScaler()

X = wine.drop(['Quality'], axis=1)
X = np.log(X)
y = wine['Quality'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

# mostre o acerto do algoritmo
print(knn.score(X_test, y_test))

print("Knn result: ", pred, "Label: ", y_test)