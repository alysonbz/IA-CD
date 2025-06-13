import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Carregar dataset
df = pd.read_csv('classificacao_ajustado.csv')
df = df.drop(columns='id')  # remover identificador
X = df.drop(columns='label')
y = df['label']

# Criar transformador logarítmico (log1p só em colunas numéricas)
log_transformer = FunctionTransformer(
    func=lambda x: np.log1p(np.clip(x, a_min=1e-5, a_max=None)), validate=True
)

# Preparar normalizações
normalizacoes = {
    'log': log_transformer,
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'none': 'passthrough'
}

# Criar pipelines com GridSearch para cada normalização
resultados = []
for nome, normalizador in normalizacoes.items():
    pipeline = Pipeline([
        ('scaler', normalizador),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': list(range(1, 21))
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5)
    grid.fit(X, y)

    melhor_k = grid.best_params_['knn__n_neighbors']
    melhor_acc = grid.best_score_
    resultados.append((nome, melhor_k, melhor_acc))
    print(f"Normalização: {nome}, Melhor k: {melhor_k}, Acurácia: {melhor_acc:.4f}")

# Ordenar pelos melhores resultados
top3 = sorted(resultados, key=lambda x: x[2], reverse=True)[:3]

# Plotar gráfico das top 3 configurações
labels = [f"{nome} (k={k})" for nome, k, _ in top3]
scores = [acc for _, _, acc in top3]

plt.figure(figsize=(8, 5))
plt.bar(labels, scores, color='blue')
plt.title("Top 3 Configurações - KNN + GridSearchCV")
plt.ylabel("Acurácia Média (CV=5)")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()
