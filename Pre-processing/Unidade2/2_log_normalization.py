import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

# 1. Print as características estatísticas do dataset wine
print("\nQuestão 1.")
print(wine.describe())

# 2. Aplique a função de normalização logarítmica na coluna Proline
wine['Proline_log'] = np.log(wine['Proline'])

# 3. Print a variância da coluna proline
print("\nQuestão 3.")
print(np.var(wine['Proline']))

# 4. Print a variância da coluna proline normalizada
print("\nQuestão 4.")
print(np.var(wine['Proline_log']))