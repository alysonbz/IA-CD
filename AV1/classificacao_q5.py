import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("classificacao_ajustado.csv")
X = df.drop("class", axis=1)
y = df["class"]

X = StandardScaler().fit_transform(X)

modelo = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(modelo, X, y, cv=5)

print("Média de acurácia:", scores.mean())
print("Desvio padrão:", scores.std())

modelo.fit(X, y)
y_pred = modelo.predict(X)

print("\nMatriz de Confusão:")
print(confusion_matrix(y, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y, y_pred))
