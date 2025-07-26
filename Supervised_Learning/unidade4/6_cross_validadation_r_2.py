from src.utils import load_sales_clean_dataset
from sklearn.linear_model import LinearRegression
# Import the necessary modules
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

# Carregar o dataset
sales_df = load_sales_clean_dataset()

# Criar X e y
X = sales_df["radio"].values.reshape(-1, 1)
y = sales_df["sales"].values

# Criar um objeto KFold com 6 divisões
kf = KFold(n_splits=6, shuffle=True, random_state=5)

# Instanciar o modelo
reg = LinearRegression()

# Calcular as pontuações de validação cruzada
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Imprimir as pontuações individuais
print("Pontuações da validação cruzada:", cv_scores)

# Imprimir a média
print("Média das pontuações:", np.mean(cv_scores))

# Imprimir o desvio padrão
print("Desvio padrão:", np.std(cv_scores))