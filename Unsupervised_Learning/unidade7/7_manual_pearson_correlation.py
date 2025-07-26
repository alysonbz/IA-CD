# Perform the necessary imports
import matplotlib.pyplot as plt
from src.utils import load_grains_dataset
import numpy as np

# Função para calcular correlação de Pearson manualmente
def pearson_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return numerator / denominator

# Carrega o dataset
grains_df = load_grains_dataset()

# Atribui as colunas de interesse
width = grains_df.iloc[:, 0]
length = grains_df.iloc[:, 1]

# Calcula a correlação de Pearson
correlation = pearson_correlation(width, length)

# Exibe a correlação
print("Correlação de Pearson:", round(correlation, 3))