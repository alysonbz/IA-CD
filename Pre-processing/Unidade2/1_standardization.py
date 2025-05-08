from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

wine = load_wine_dataset()

#scaler = StandardScaler()
scaler = MinMaxScaler()

X = wine.drop(['Quality'],axis=1)
#X = np.log(X)
X = scaler.fit_transform(X)
y = wine['Quality'].values

# divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)

# Aplique a função fit do knn
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

# mostre o acerto do algoritmo
print(knn.score(X_test, y_test))
print(f"knn result: {pred} \n label: {y_test}")