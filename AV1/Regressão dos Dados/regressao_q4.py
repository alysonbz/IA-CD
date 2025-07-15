import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Carregar o dataset ajustado
df = pd.read_csv('regressao_ajustado.csv')

# Separar as variáveis independentes (X) e a dependente (y)
X = df.drop(columns='price')
y = df['price']

# Instanciar o modelo Lasso
model = Lasso(alpha=0.1)
model.fit(X, y)

# Obter os coeficientes
coeficients = pd.Series(model.coef_, index=X.columns)

# Ordenar os coeficientes do menor para o maior
coeficients_sorted = coeficients.sort_values()

# Plotar o gráfico de barras dos coeficientes
plt.figure(figsize=(10, 6))
coeficients_sorted.plot(kind='barh', color='skyblue')
plt.title('Importância dos Atributos - Modelo Lasso')
plt.xlabel('Coeficientes')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Mostrar os coeficientes numericamente
print(coeficients_sorted)

# ---------------------- 📊 Discussão dos Resultados --------------------------
# As variáveis “room_type_Shared room” (-140.39) e “room_type_Private room” (-110.91)
# possuem os maiores impactos negativos no preço, indicando que quartos compartilhados
# ou privados são significativamente mais baratos que apartamentos inteiros
# (que é a categoria de referência, não aparece no modelo).

# O bairro “Manhattan” (+75.56) tem forte impacto positivo, ou seja,
# os preços são mais elevados nessa região.

# Brooklyn também tem impacto positivo (+19.88), mas bem menor que Manhattan.

# O modelo revela que os atributos “localização” (bairro) e “tipo de acomodação”
# são os mais relevantes para prever preços no Airbnb.

# As demais variáveis quantitativas (mínimo de noites, número de reviews, etc.)
# têm impacto quase nulo no modelo linear, ou seja, não contribuem de forma relevante.
# -----------------------------------------------------------------------------
