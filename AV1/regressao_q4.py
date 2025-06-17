import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 1. Carregar dataset ajustado
df = pd.read_csv('dataset/regressao_ajustado.csv')
X = df.drop(columns=['cnt'])
y = df['cnt']
colunas = X.columns

# 2. Treinar modelo Lasso com mais iterações para garantir convergência
modelo = Lasso(alpha=0.1, max_iter=10000)
modelo.fit(X, y)

# 3. Obter coeficientes
coeficientes = modelo.coef_

# 4. Criar DataFrame com os coeficientes
df_coef = pd.DataFrame({
    'Atributo': colunas,
    'Coeficiente': coeficientes
})
df_coef['Importância Absoluta'] = df_coef['Coeficiente'].abs()
df_coef_ordenado = df_coef.sort_values(by='Importância Absoluta', ascending=False)

# 5. Identificar atributos mais relevantes (coeficiente ≠ 0)
atributos_relevantes = df_coef[df_coef['Coeficiente'] != 0]

# 6. Plotar gráfico de importância
plt.figure(figsize=(12, 6))
plt.bar(df_coef_ordenado['Atributo'], df_coef_ordenado['Coeficiente'], color='teal')
plt.xticks(rotation=45)
plt.title("Importância dos Atributos segundo Lasso (Coeficientes)")
plt.xlabel("Atributos")
plt.ylabel("Coeficiente")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.show()

# 7. Exibir e comentar
print("\nAtributos relevantes (coeficiente diferente de zero):")
print(atributos_relevantes.to_string(index=False))

print("\nComentário:")
print("- O modelo Lasso atribui coeficiente zero a alguns atributos, realizando seleção automática de variáveis.")
print("- Atributos com maior valor absoluto de coeficiente têm maior impacto na predição.")
print("- É possível reduzir a dimensionalidade mantendo apenas os atributos relevantes.")
