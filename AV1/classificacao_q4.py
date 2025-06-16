import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Carrega dataset
df = pd.read_csv('dataset/classificacao_ajustado.csv')
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Normalizações
normalizacoes = {
    'log': FunctionTransformer(np.log1p, validate=True),
    'minmax': MinMaxScaler(),
    'standard': StandardScaler(),
    'none': FunctionTransformer(lambda x: x, validate=True)
}

# Pipeline
pipe = Pipeline([
    ('normalizacao', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# Param grid
param_grid = {
    'normalizacao': list(normalizacoes.values()),
    'knn__n_neighbors': list(range(1, 21))
}

# GridSearch
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', return_train_score=True, n_jobs=-1)
grid.fit(X_train, y_train)

# Resultados
resultados = pd.DataFrame(grid.cv_results_)

# Top 3
top3 = resultados.sort_values(by='mean_test_score', ascending=False).head(3)
print("Top 3 configurações:")
print(top3[['param_normalizacao', 'param_knn__n_neighbors', 'mean_test_score']])

# Gráfico: Acurácia vs K para cada das 3 melhores normalizações
plt.figure(figsize=(10, 6))
for norm in top3['param_normalizacao'].unique():
    nome = type(norm).__name__
    if isinstance(norm, FunctionTransformer) and norm.func == np.log1p:
        nome = 'log'
    elif isinstance(norm, FunctionTransformer) and norm.func.__name__ == '<lambda>':
        nome = 'none'

    subset = resultados[resultados['param_normalizacao'] == norm]
    plt.plot(subset['param_knn__n_neighbors'], subset['mean_test_score'], marker='o', label=f'{nome}')

plt.title('Acurácia média vs K para Top 3 Normalizações')
plt.xlabel('K')
plt.ylabel('Acurácia média (cross-validation)')
plt.legend(title='Normalização')
plt.grid(True)
plt.tight_layout()
plt.show()
