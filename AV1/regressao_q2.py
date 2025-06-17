import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Carregar o dataset ajustado
df = pd.read_csv("regressao_ajustado.csv")

# 2. Selecionar a variável mais correlacionada com a variável-alvo
target = 'Y house price of unit area'
correlations = df.corr(numeric_only=True)[target].drop(target).sort_values(ascending=False)
top_feature = correlations.index[0]
print(f"✅ Feature mais correlacionada com '{target}': {top_feature} (correlação = {correlations[0]:.3f})")

# 3. Preparar dados para regressão
X = df[[top_feature]].values  # variável explicativa
y = df[target].values         # variável alvo

# 4. Aplicar Regressão Linear
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 6. Calcular métricas de desempenho
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(f"📊 RMSE: {rmse:.4f}")
print(f"📈 R²: {r2:.4f}")

# 7. Comentário sobre os resultados
print("\n📌 Comentário:")
if r2 > 0.7:
    print(f"A variável '{top_feature}' tem uma boa explicação sobre a variação do preço, com R² = {r2:.2f}.")
elif r2 > 0.4:
    print(f"A variável '{top_feature}' tem uma explicação moderada sobre o preço. O modelo captura parcialmente a variação.")
else:
    print(f"A variável '{top_feature}' explica pouco da variação no preço (R² = {r2:.2f}). Considere múltiplas variáveis para melhor desempenho.")
# 5. Plotar a reta de regressão
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.5, label='Dados reais')
plt.plot(X, y_pred, color='red', label='Reta de regressão')
plt.xlabel(top_feature)
plt.ylabel(target)
plt.title("Regressão Linear Simples")
plt.legend()
plt.grid(True)
plt.show()


