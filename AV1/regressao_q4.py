import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Carregar dataset ajustado
df = pd.read_csv("regressao_ajustado.csv")
X = df.drop(columns=["G3"])
y = df["G3"]

# Aplicar Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Obter coeficientes
coef = pd.Series(lasso.coef_, index=X.columns)

# Filtrar atributos relevantes
coef_nao_zero = coef[coef != 0].sort_values(key=abs, ascending=False)

# Exibir
print("Atributos mais relevantes segundo Lasso:\n")
print(coef_nao_zero)

# Plotar gráfico de importância
plt.figure(figsize=(10, 6))
coef_nao_zero.plot(kind="bar")
plt.title("Importância dos Atributos (Coeficientes Lasso)")
plt.ylabel("Peso do Coeficiente")
plt.xlabel("Atributo")
plt.grid(True)
plt.tight_layout()
plt.savefig("regressao_q4_importancia.png")
plt.show()