import numpy as np
from scipy.stats import describe

from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()
describe(load_wine_dataset())

pd.set_option('display.max_columns', None)

#print as caractéristicas estatísticas do dataset wine
print(wine.describe())

## Aplique a função de nomarlização logarítmica na coluna Proline
wine["Proline"] = np.log("Proline_log")

# Print a variância da coluna proline
print(np.var["Proline_log"])

# print a variância da coluna proline normalizada
print(np.var(wine['Proline_log']))
