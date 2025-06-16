import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Carrega o dataset ajustado
df = pd.read_csv("regressao_ajustado.csv")

# Regressão com a variável mais correlacionada (ex: Length3)
X = df[['Length3']]
y = df['Weight']

# Modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Previsões
y_pred = modelo.predict(X)

# Métricas
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

# Gráfico
plt.scatter(X, y, label="Real")
plt.plot(X, y_pred, color='red', label="Previsão")
plt.xlabel("Length3 (normalizado)")
plt.ylabel("Weight")
plt.title("Regressão Linear Simples")
plt.legend()
plt.savefig("regressao_linear_plot.png")
plt.close()
