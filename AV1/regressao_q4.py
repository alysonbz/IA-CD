import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 0. Carregando dataset
df = pd.read_csv("regressao_ajustado.csv")
print(df.head(), '\n')
print(df.shape)

X = df[['enginesize','carheight','horsepower']]
y = df["price"].values

# Padronizar os dados (importante para Lasso)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Treinar o modelo Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# 1. Visualizar os coeficientes de lasso
coef = lasso.coef_
atributos =  X.columns

# Criando um DataFrame com coeficientes para facilitar análise
df_coef = pd.DataFrame({'Atributo': atributos, 'Coeficiente': coef})
df_coef['Importância Absoluta'] = df_coef['Coeficiente'].abs()


# Ordenar pelos coeficientes absolutos (maior importância)
df_coef_sorted = df_coef.sort_values(by='Importância Absoluta', ascending=False)

print("Coeficientes do modelo Lasso:")
print(df_coef_sorted)

# 3. Plotar gráfico de importância dos atributos
plt.figure(figsize=(10,6))
plt.barh(df_coef_sorted['Atributo'], df_coef_sorted['Importância Absoluta'])
plt.xlabel('Importância Absoluta do Coeficiente')
plt.title('Importância dos Atributos segundo Lasso')
plt.gca().invert_yaxis()  # inverte eixo para o maior ficar em cima
plt.show()