# 7
# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_grains_dataset

# Função que calcula a correlação de Pearson manualmente
def pearson_correlation(x, y):
    x = np.array(x)
    y = np.array(y)
    x_mean = x.mean()
    y_mean = y.mean()
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return numerator / denominator

# Carregar os dados
grains_df = load_grains_dataset()

# Atribuir colunas
width = grains_df.iloc[:, 0]
length = grains_df.iloc[:, 1]

# Calcular correlação de Pearson
correlation = pearson_correlation(width, length)

# Exibir correlação
print("Correlação de Pearson:", correlation)
