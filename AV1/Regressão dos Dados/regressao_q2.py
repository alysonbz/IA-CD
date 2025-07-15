import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Carregar o dataset já tratado e ajustado
df = pd.read_csv('regressao_ajustado.csv')

# Calcular a correlação para identificar a feature mais relacionada com o preço
correlation = df.corr()['price'].sort_values(ascending=False)

# Selecionar a variável (feature) mais correlacionada, excluindo o próprio preço
top_feature = correlation.drop('price').idxmax()

# Separar a feature selecionada (X) e a variável alvo (y)
X_feature = df[[top_feature]].values
y_target = df['price'].values

# Instanciar o modelo de Regressão Linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_feature, y_target)

# Realizar as previsões
y_pred = model.predict(X_feature)

# Calcular as métricas de desempenho: RMSE e R²
rmse = np.sqrt(mean_squared_error(y_target, y_pred))
r2 = r2_score(y_target, y_pred)

# Plotando a reta de regressão junto com os dados
plt.figure(figsize=(8,6))
plt.scatter(X_feature, y_target, color='blue', alpha=0.5, label='Dados')
plt.plot(X_feature, y_pred, color='red', linewidth=2, label='Reta de Regressão')
plt.title(f'Regressão Linear - Feature: {top_feature}')
plt.xlabel(top_feature)
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar os resultados
print(f'Feature mais correlacionada: {top_feature}')
print(f'R²: {r2:.4f}')
print(f'RMSE: {rmse:.2f}')

# ---------------------- COMENTÁRIO SOBRE OS RESULTADOS ----------------------
# O modelo de regressão linear simples apresentou um desempenho bem fraco.
# O R² foi de aproximadamente 0,027, ou seja, apenas 2,7% da variação dos preços
# é explicada por essa única variável.
# 
# Além disso, o RMSE ficou em torno de 237, indicando um erro médio bem alto 
# na previsão dos preços. Isso mostra que o preço dos imóveis não depende apenas
# de uma variável, mas sim da combinação de vários fatores.
#
# Conclusão: O modelo linear simples não é suficiente. É necessário usar
# modelos com múltiplas variáveis para obter melhores resultados.
# -----------------------------------------------------------------------------
