import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# CARREGAR DATASET
df = pd.read_csv('dataset_tratado.csv')

# SEPARAÇÃO X E y
X = df.drop('Revenue', axis=1)
y = df['Revenue']


# NORMALIZAÇÃO LOG
#A NORMALIZAÇÃO LOG FOI ESCOLHIDA APÓS APRESENTAR MELHORES RESULTADOS
X = np.log1p(X)

# MODELO KNN
knn = KNeighborsClassifier()

# VALIDAÇÃO CRUZADA
scores = cross_val_score(knn, X, y, cv=5)

print('VALIDAÇÃO CRUZADA')
print('Scores:', scores)
print(f'Média da acurácia: {scores.mean():.4f}')


# GRID SEARCH
parametros = {
    'n_neighbors': range(1, 21)
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    parametros,
    cv=5
)

grid.fit(X, y)

print('\n====================================')
print('GRID SEARCH')
print('====================================')

print('Melhor k:', grid.best_params_)

print(f'Melhor acurácia: {grid.best_score_:.4f}')

# COMPARAÇÃO FINAL
print('\n====================================')
print('COMPARAÇÃO')
print('====================================')
print('Melhor k encontrado visualmente: 18')
print('Melhor k encontrado pelo Grid Search:',
      grid.best_params_['n_neighbors'])