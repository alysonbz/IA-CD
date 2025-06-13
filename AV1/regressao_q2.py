import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Carregar o dataset ajustado
df = pd.read_csv('regressao_ajustado.csv')

# Identificar a feature mais correlacionada com MEDV
correlations = df.corr()[['MEDV']].sort_values(by='MEDV', ascending=False)
top_feature = correlations.index[1]  # O primeiro é o próprio MEDV
print(f"\nFeature mais correlacionada com MEDV: {top_feature}")

# Preparar dados para regressão
X = df[[top_feature]].values
y = df['MEDV'].values

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nMétricas do Modelo:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Plotar resultados
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Reta de regressão')
plt.title(f'Regressão Linear Simples: {top_feature} vs MEDV')
plt.xlabel(top_feature)
plt.ylabel('MEDV')
plt.legend()
plt.grid(True)
plt.savefig('regressao_simples.png')
plt.close()

print("\nGráfico salvo como 'regressao_simples.png'")