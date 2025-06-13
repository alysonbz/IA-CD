import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 0. Carregando dataset
df = pd.read_csv("regressao_ajustado.csv")
print(df.head(), '\n')
print(df.shape)


# 1. Aplique Regressão Linear (`sklearn`).
X = df["enginesize"].values.reshape(-1, 1)
y = df["price"].values

modelo = LinearRegression()
modelo.fit(X, y)
predictions = modelo.predict(X)
print(predictions)

# 2. Plote a reta de regressão (feature mais correlacionada).
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("enginesize ")
plt.ylabel("price ")

# Display the plot
plt.show()

# 3. Calcule RMSE e R².
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# 4.comentario sobre o resultado
print(df.describe())
# O rmse esta muito bom porque a media é 13276 e o erro é 3870
# e o R^2 está proximo de 1