import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Carregar o dataset ajustado
df = pd.read_csv('dataset/regressao_ajustado.csv')

# 2. Identificar a feature mais correlacionada com a variável-alvo
correlacoes = df.corr(numeric_only=True)['cnt'].drop('cnt')
feature_mais_corr = correlacoes.abs().idxmax()
print(f"Feature mais correlacionada com 'cnt': {feature_mais_corr} (correlação = {correlacoes[feature_mais_corr]:.2f})")

# 3. Definir X e y para regressão linear simples
X = df[[feature_mais_corr]]
y = df['cnt']

# 4. Aplicar Regressão Linear
modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

# 5. Plotar a reta de regressão
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', linewidth=2, label='Reta de Regressão')
plt.title(f"Regressão Linear Simples: {feature_mais_corr} vs cnt")
plt.xlabel(feature_mais_corr)
plt.ylabel("cnt")
plt.legend()
plt.grid(True)
plt.show()

# 6. Calcular métricas
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")


# 7. Comentário sobre os resultados
print("\nComentário:")
if r2 > 0.7:
    print("- A variável explicativa tem uma forte relação com a variável-alvo.")
elif r2 > 0.4:
    print("- A variável explicativa tem uma relação moderada com a variável-alvo.")
else:
    print("- A variável explicativa tem uma fraca relação com a variável-alvo. Um modelo multivariado pode ser mais adequado.")
