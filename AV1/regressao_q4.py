import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados
df = pd.read_csv('housing.csv')
X = df.drop('MEDV', axis=1)
y = df['MEDV']
feature_names = X.columns

# 2. Padronizar as variáveis independentes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Treinar o modelo Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# 4. Coeficientes e atributos relevantes
coeficientes = lasso.coef_
atributos_relevantes = [
    (feature, coef) for feature, coef in zip(feature_names, coeficientes) if coef != 0
]

# 5. Ordenar por importância absoluta
atributos_relevantes.sort(key=lambda x: abs(x[1]), reverse=True)
nomes, valores = zip(*atributos_relevantes)

# 6. Plotar gráfico de importância
plt.figure(figsize=(12, 6))
cores = ['green' if v > 0 else 'red' for v in valores]
plt.bar(nomes, valores, color=cores)
plt.title('Importância dos Atributos segundo Lasso Regression', fontsize=14)
plt.ylabel('Coeficiente', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
