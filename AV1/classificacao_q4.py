# 1. Use KNeighborsClassifier da Sklearn.

import sklearn.model_selection as model_selection


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import GridSearchCV

log_transformer = FunctionTransformer(np.log1p, validate=True)
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

p = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

##

# 2. Teste valores de k entre 1 e 20 e as normalizações com GridSearchCV.

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split



# Transformações
log_transformer = FunctionTransformer(np.log1p, validate=True)
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()), 
    ("knn", KNeighborsClassifier())
])

# Parâmetro (1, 20)
parametro = {
    'scaler': [standard_scaler, minmax_scaler, log_transformer],
    'knn__n_neighbors': list(range(1, 21))
}

# GridSearchCV
grid = GridSearchCV(pipe, parametro, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# Resultados
print("Melhor combinação:")
print(grid.best_params_)
print("Melhor score:")
print(grid.best_score_)


##

# As 3 melhores combinações

results = pd.DataFrame(grid.cv_results_)
results = results.sort_values(by="mean_test_score", ascending=False)

top3 = results.head(3)
print(top3[['mean_test_score', 'param_scaler', 'param_knn__n_neighbors']])

##

# 3. Plote gráfico dos resultados com as 3 melhores configurações.

plt.figure(figsize=(10,6))
for i in range(3):
    label = f"{top3.iloc[i]['param_scaler'].__class__.__name__} + k={top3.iloc[i]['param_knn__n_neighbors']}"
    plt.plot([i+1], [top3.iloc[i]['mean_test_score']], marker='o', label=label)

plt.title("Top 3 Configurações do GridSearchCV")
plt.ylabel("Acurácia média (cv=5)")
plt.xticks([])
plt.ylim(0.5, 1)
plt.legend()
plt.grid(True)
plt.show()

