import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# Carregar dados
df = pd.read_csv("classificacao_ajustado.csv")

# Selecionar colunas numéricas úteis
X = df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']]
y = df['class']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline e parâmetros
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
    'knn__n_neighbors': range(1, 21)
}

# GridSearchCV
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# Top 3 configurações
results = pd.DataFrame(grid.cv_results_)
results['scaler'] = results['param_scaler'].apply(lambda x: type(x).__name__)
top3 = results.sort_values(by='mean_test_score', ascending=False).head(3)

print("\nTop 3 melhores configurações:")
print(top3[['params', 'mean_test_score']])

# Gráfico das top 3
plt.figure(figsize=(8, 5))
for _, row in top3.iterrows():
    scaler = type(row['params']['scaler']).__name__
    mask = results['scaler'] == scaler
    subset = results[mask]
    plt.plot(subset['param_knn__n_neighbors'], subset['mean_test_score'], label=scaler)

plt.xlabel('k')
plt.ylabel('Acurácia Média (CV)')
plt.title('Top 3 Configurações - KNN')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
