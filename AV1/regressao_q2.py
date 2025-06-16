import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Carregar dados pré-processados
df = pd.read_csv(r"C:\Users\jsslu\OneDrive\Área de Trabalho\UFC\Inteligencia Artificial\git\IA-CD\AV1\datasets\regressao_ajustado.csv")

# Identificar a feature mais correlacionada com a variável-alvo
correlacao = df.corr()['charges'].drop('charges')
feature_mais_correlacionada = correlacao.abs().idxmax()
print(f"Feature mais correlacionada: {feature_mais_correlacionada}")

# Separar variáveis independentes e alvo
X = df[[feature_mais_correlacionada]]
y = df['charges']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliação
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\nRMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Plot
plt.scatter(X_test, y_test, color='blue', label='Real')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Previsão')
plt.xlabel(feature_mais_correlacionada)
plt.ylabel('charges')
plt.title('Regressão Linear Simples')
plt.legend()
plt.show()
