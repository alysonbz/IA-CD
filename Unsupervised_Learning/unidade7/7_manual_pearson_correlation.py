# Perform the necessary imports
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_grains_dataset


def pearson_correlation(x,y):
    # Convertendo para arrays numéricos caso sejam séries do pandas
    x = np.array(x)
    y = np.array(y)

    # Calculando as médias de x e y
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculando a covariância e as variâncias (numerador e denominador)
    num = np.sum((x - mean_x) * (y - mean_y))
    den = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))

    # Retorna o coeficiente de correlação de Pearson
    return num / den


grains_df = load_grains_dataset()


# Assign the 0th column of grains: width
width = grains_df['0']

# Assign the 1st column of grains: length
length = grains_df['1']

# Calculate the Pearson correlation
correlation = pearson_correlation(width,length)

# Display the correlation
print(correlation)

