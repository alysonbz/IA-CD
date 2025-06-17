# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
import matplotlib.pyplot as plt

# Dataset
df = pd.read_csv('C:/Users/xulia/IA-CD/IA-CD/AV1/classificacao_ajustado.csv')
df['Medicamento'] = df['Medicamento'].astype(int)
X = pd.get_dummies(df.drop(columns=['Medicamento']), drop_first=True).astype(float)
y = df['Medicamento']

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Normalizações
scalers = {
    'none': FunctionTransformer(validate=True),
    'standard': StandardScaler(),
    'minmax': MinMaxScaler()
}

# Preparar busca com GridSearch
param_grid = {
    'scaler': list(scalers.values()),
    'knn__n_neighbors': list(range(1, 21))
}

pipe = Pipeline([
    ('scaler', StandardScaler()),  # valor inicial arbitrário
    ('knn', KNeighborsClassifier())
])

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Resultados
results = pd.DataFrame(grid.cv_results_).sort_values(by='mean_test_score', ascending=False)
top3 = results[['param_knn__n_neighbors', 'param_scaler', 'mean_test_score']].head(3)

# Gráfico das 3 melhores
plt.figure(figsize=(8, 5))
plt.bar(
    x=[f"k={row['param_knn__n_neighbors']}\n{row['param_scaler'].__class__.__name__}" for _, row in top3.iterrows()],
    height=top3['mean_test_score']
)
plt.title("Top 3 configurações - GridSearchCV KNN")
plt.ylabel("Acurácia Média (CV)")
plt.tight_layout()
plt.show()

top3.reset_index(drop=True, inplace=True)
print("Top 3 configurações -KNN com GridSearchCV")
print(top3)
