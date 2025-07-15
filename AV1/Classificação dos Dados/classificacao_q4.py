import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# 🔧 Classe para transformação logarítmica
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.shift_ = np.abs(np.min(X, axis=0)) + 1e-6  # evitar log de zero ou negativo
        return self

    def transform(self, X):
        return np.log(X + self.shift_)

# ✅ Pipeline genérico
pipeline = Pipeline([
    ('scaler', 'passthrough'),  # Placeholder que será substituído no param_grid
    ('knn', KNeighborsClassifier(metric='chebyshev'))
])

# ✅ Parâmetros para GridSearch
param_grid = {
    'scaler': [LogTransformer(), MinMaxScaler(), StandardScaler()],
    'knn__n_neighbors': list(range(1, 21))  # k de 1 a 20
}

# 🚀 Rodar GridSearchCV
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# ⏳ Ajustar
grid.fit(X.to_numpy(), y_binary)

# 📊 Coletar os resultados
results = pd.DataFrame(grid.cv_results_)

# 🏷️ Adicionar coluna com o nome do scaler (classe)
results['scaler_name'] = results['param_scaler'].apply(lambda x: type(x).__name__)

# 🥇 Mostrar as 5 melhores combinações
print("Top 5 melhores configurações:")
print(results[['scaler_name', 'param_knn__n_neighbors', 'mean_test_score']].sort_values(by='mean_test_score', ascending=False).head(5))

# 🎨 Plotar os 3 melhores scalers
top3_scaler_names = results.groupby('scaler_name')['mean_test_score'].max().sort_values(ascending=False).head(3).index.tolist()

plt.figure(figsize=(10, 6))
for scaler_name in top3_scaler_names:
    mask = results['scaler_name'] == scaler_name
    plt.plot(
        results[mask]['param_knn__n_neighbors'],
        results[mask]['mean_test_score'],
        marker='o',
        label=scaler_name
    )
plt.title('Top 3 Normalizações - Acurácia vs k')
plt.xlabel('Valor de k')
plt.ylabel('Acurácia Média (Cross-Validation)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Segunda parte - repetida com outra abordagem
normalizations = {
    'log': LogTransformer(),
    'minmax': MinMaxScaler(),
    'standard': StandardScaler(),
}

# Montar pipelines para cada normalização
pipelines = []
for name, scaler in normalizations.items():
    pipe = Pipeline([
        ('scaler', scaler),
        ('knn', KNeighborsClassifier(metric='chebyshev'))
    ])
    pipelines.append((name, pipe))

# Montar param_grid para GridSearch
param_grid = []
for name, pipe in pipelines:
    param_grid.append({
        'scaler': [normalizations[name]],
        'knn__n_neighbors': list(range(1, 21))
    })

# Juntar os dados novamente
X_array = np.array(X)
y_array = np.array(y_binary)

# Rodar GridSearchCV com novo pipeline
grid = GridSearchCV(
    estimator=Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(metric='chebyshev'))]),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_array, y_array)

# Organizar os melhores resultados
df_results = pd.DataFrame(grid.cv_results_).sort_values(by='mean_test_score', ascending=False)

# Pegar as 3 melhores
top3 = df_results.head(3)
print("\nTop 3 resultados finais:")
print(top3[['param_scaler', 'param_knn__n_neighbors', 'mean_test_score']])
