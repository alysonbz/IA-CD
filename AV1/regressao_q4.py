import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Carregar dados pré-processados
df = pd.read_csv(r"C:\Users\jsslu\OneDrive\Área de Trabalho\UFC\Inteligencia Artificial\git\IA-CD\AV1\datasets\regressao_ajustado.csv")
X = df.drop('charges', axis=1)
y = df['charges']

# Treinar modelo Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Coeficientes
coef = pd.Series(lasso.coef_, index=X.columns)
coef_nonzero = coef[coef != 0].sort_values(ascending=False)

print("\nAtributos mais relevantes (coeficientes diferentes de zero):")
print(coef_nonzero)

# Gráfico de importância
plt.figure(figsize=(10, 6))
coef_nonzero.plot(kind='bar')
plt.title('Importância dos Atributos - Lasso Regression')
plt.ylabel('Coeficiente')
plt.xlabel('Atributos')
plt.tight_layout()
plt.show()
