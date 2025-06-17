import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv('classificacao_ajustado.csv')
X = df.drop(columns=['class_e'])
y = df['class_e']

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizações
scalers = {
    'MinMax': MinMaxScaler(),
    'Standard': StandardScaler(),
}

# Parâmetros
param_grid = {'n_neighbors': list(range(1, 21))}

resultados = {}

for nome, scaler in scalers.items():
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5)
    grid.fit(X_train_s, y_train)
    resultados[nome] = grid

# Plotar melhores resultados
for nome, grid in resultados.items():
    print(f"{nome} - Melhor K: {grid.best_params_['n_neighbors']}, Acurácia: {grid.best_score_:.4f}")

scores = [grid.best_score_ for grid in resultados.values()]
plt.bar(resultados.keys(), scores)
plt.title("Melhores configurações - Acurácia Média")
plt.ylabel("Acurácia")
plt.show()
