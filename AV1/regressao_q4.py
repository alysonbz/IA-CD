import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Carregar dataset ajustado
df = pd.read_csv('regressao_ajustado.csv')

X = df.drop('Weight', axis=1)
y = df['Weight']

# Treinar Lasso
lasso = Lasso()
lasso.fit(X, y)

# Coeficientes
coef = pd.Series(lasso.coef_, index=X.columns)

# Mostrar coeficientes relevantes
print("Coeficientes Lasso:")
print(coef)

# Plotar importância
coef.plot(kind='bar')
plt.title('Importância dos atributos pelo Lasso')
plt.show()
