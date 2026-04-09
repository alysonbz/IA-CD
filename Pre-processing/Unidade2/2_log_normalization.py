import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

# Print as características estatísticas do dataset wine
print("\nQuestão.")
print(wine.describe())

# Aplique a função de normalização logarítmica na coluna Proline
wine['Proline_ln'] = np.log(wine['Proline'])

# Print a variância da coluna proline
print("\nQuestão.")
print(np.var(wine['Proline']))

# Print a variância da coluna proline normalizada
print("\nQuestão.")
print(np.var(wine['Proline_ln']))