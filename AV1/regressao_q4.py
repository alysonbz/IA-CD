import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Carrega os dados
df = pd.read_csv("regressao_ajustado.csv")
X = df.drop(columns=['Weight'])
y = df['Weight']

# Treina o modelo Lasso
modelo = Lasso(alpha=0.1)
modelo.fit(X, y)

# Coeficientes
coeficientes = pd.Series(modelo.coef_, index=X.columns)
print("Coeficientes da Lasso:\n", coeficientes)

# Gráfico de importância
coeficientes.sort_values().plot(kind='barh')
plt.title("Importância dos Atributos (Lasso)")
plt.xlabel("Coeficiente")
plt.tight_layout()
plt.savefig("lasso_importancia.png")
plt.close()
