import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

# 1. Print as características estatísticas do dataset wine
print("\nQuestão 1.")
print(wine.describe())
# describe() mostra resumo estatístico: média, desvio padrão, mínimo, máximo, etc.

# 2. Aplique a função de normalização logarítmica na coluna Proline
wine['Proline_ln'] = np.log(wine['Proline'])
# np.log() aplica log natural para reduzir a escala dos valores

# 3. Print a variância da coluna proline
print("\nQuestão 3.")
print(np.var(wine['Proline']))
# np.var() calcula o quanto os valores estão espalhados (variância)

# 4. Print a variância da coluna proline normalizada
print("\nQuestão 4.")
print(np.var(wine['Proline_ln']))
# calcula a variância após o log para comparar a dispersão dos dados