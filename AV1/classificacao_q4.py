import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("classificacao_ajustado.csv")
X = df.drop("class", axis=1)
y = df["class"]

scalers = {
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler(),
    "Nenhum": None
}

resultados = []

for nome, scaler in scalers.items():
    X_norm = scaler.fit_transform(X) if scaler else X
    model = KNeighborsClassifier()
    grid = GridSearchCV(model, {"n_neighbors": list(range(1, 21))}, cv=5)
    grid.fit(X_norm, y)
    melhores = grid.best_params_['n_neighbors']
    acc = grid.best_score_
    resultados.append((nome, acc, melhores))

# Plot dos resultados
labels = [x[0] for x in resultados]
scores = [x[1] for x in resultados]

plt.bar(labels, scores)
plt.title("Acurácia por Normalização")
plt.ylabel("Acurácia")
plt.savefig("gridsearch_knn_resultados.png")
