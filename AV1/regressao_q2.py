import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Carregar o dataset ajustado
df = pd.read_csv("regressao_ajustado.csv")

# Separar variável preditora (G2) e alvo (G3)
X = df[["G2"]]
y = df["G3"]

# Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# Fazer previsões
y_pred = model.predict(X)

# Calcular métricas
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# Exibir resultados
print(f"Coeficiente angular (inclinação): {model.coef_[0]:.4f}")
print(f"Intercepto: {model.intercept_:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Plotar reta de regressão
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label="Dados reais")
plt.plot(X, y_pred, color='red', label="Reta de regressão")
plt.xlabel("G2 (padronizado)")
plt.ylabel("G3 (padronizado)")
plt.title("Regressão Linear Simples - G2 vs G3")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("regressao_q2_plot.png")
plt.show()