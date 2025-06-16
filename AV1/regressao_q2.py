# 1. Aplique Regressão Linear (sklearn).

from sklearn.linear_model import LinearRegression

X = df2_scaled.drop("csMPa", axis=1)
y = df2_scaled["csMPa"]

model = LinearRegression()
model.fit(X, y)

print("B1:", model.coef_)
print("B0:", model.intercept_)
print("Treinamento:", model.score(X, y))

##

# 2. Plote a reta de regressão (feature mais correlacionada).

X = df2_scaled[["age"]]  
y = df2_scaled["csMPa"]

# Treina a regressão linear simples
model = LinearRegression()
model.fit(X, y)

x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_vals = model.predict(x_vals)

# Pontos
plt.scatter(X, y, color='blue', label='Dados')

# Reta de regressão
plt.plot(x_vals, y_vals, color='red', label='Reta de regressão')
plt.xlabel("Age")
plt.ylabel("csMPa")
plt.title("Reta de Regressão Linear - Age vs csMPa")
plt.legend()
plt.grid(True)
plt.show()

##

# 3. Calcule RMSE e R².

from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
print("R2: ", r2)
print("RMSE:", rmse) # Errro quadrático médio

##

# 4. Comente sobre os resultados.

# Perante a análise notou-se que os  valores de B0 (intercepeto) são bem  próximos de zero, onde indique que os dados estão bem padronizados pelo método StandardScaler.

# B1 por sua vez mostra influência de cada variável para com a resistência do concreto.

# Treinamento de 0.6155 (61.55%), para essa análise esse valor é consideravelmente bom, pois significa que ese valor reprsenta a variação da resistencia (csMPa) que é explicada pelo modelo.

# O desempenho desse modelo com com R2 próximo de 0.1081 que corresponde apenas 10.8% de variação dos dados, demonstrando que a variável age não é um bom influenciador para csMPa.

# RMSE de 0.9443, bem próximo de 1, indicando que, esse modelo erra cerca de 1 desvio padrão. Nesse contexto, esse erro nesse valor é considerado alto, em relação ao baixo valor de R2.



