import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 1. Carregar o dataset ajustado
df = pd.read_csv("regressao_ajustado.csv")
target = 'Y house price of unit area'
X = df.drop(columns=[target])
y = df[target]

# 2. Treinar modelo Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# 3. Obter coeficientes
coeficientes = pd.Series(lasso.coef_, index=X.columns)

# 4. Visualizar os coeficientes diferentes de zero
coef_nao_nulos = coeficientes[coeficientes != 0].sort_values(ascending=False)
print("✅ Atributos mais relevantes (coeficientes ≠ 0):\n")
print(coef_nao_nulos)

# 5. Plotar gráfico de importância
plt.figure(figsize=(10, 6))
coef_nao_nulos.plot(kind='barh')
plt.title("Importância dos Atributos segundo a Lasso")
plt.xlabel("Coeficiente")
plt.ylabel("Atributos")
plt.grid(True)
plt.gca().invert_yaxis()  # Colocar maior valor no topo
plt.tight_layout()
plt.show()

# 6. Discussão automática
print("\n📌 Discussão:")
if len(coef_nao_nulos) == 0:
    print("O modelo Lasso com alpha = 0.1 não considerou nenhum atributo relevante. Tente diminuir o alpha.")
else:
    print(f"O modelo Lasso identificou {len(coef_nao_nulos)} atributos relevantes para prever o preço de imóveis.")
    print("Os atributos com maiores coeficientes (positivos ou negativos) têm maior influência no modelo.")
