import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar dataset
df = pd.read_csv('/root/.cache/kagglehub/datasets/vipullrathod/fish-market/versions/1/Fish.csv')

# Remover coluna categórica (Species) para cálculo de correlação
df_num = df.drop('Species', axis=1)

# Verificar correlação com a variável alvo Weight
correlation = df_num.corr()['Weight'].sort_values(ascending=False)
feature = correlation.index[1]  # Pega a feature mais correlacionada (exceto o próprio Weight)
print(f"Feature mais correlacionada: {feature}")

# Preparar dados para regressão
X = df[[feature]].values  # Feature mais correlacionada (ex: Length3)
y = df['Weight'].values

# Ajustar modelo de regressão linear simples
model = LinearRegression()
model.fit(X, y)

# Fazer predições
y_pred = model.predict(X)

# Métricas (corrigido para versões antigas do sklearn)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Plotar reta de regressão
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Regressão Linear')
plt.xlabel(feature)
plt.ylabel('Weight')
plt.title('Regressão Linear Simples')
plt.legend()
plt.show()
