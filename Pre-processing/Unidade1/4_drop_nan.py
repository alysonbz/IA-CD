from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.
import pandas as pd
from src.utils import load_volunteer_dataset

# Carregar o dataset
volunteer = load_volunteer_dataset()

# Remover colunas com todos os valores NaN
volunteer_sem_colunas_nan = volunteer.dropna(axis=1, how='all')

# Remover linhas com qualquer valor NaN
volunteer_limpo = volunteer_sem_colunas_nan.dropna()

# Mostrar a contagem de valores NaN por coluna (deve ser tudo zero)
print("Contagem de valores NaN por coluna:")
print(volunteer_limpo.isna().sum())

# Mostrar o novo shape do dataframe
print("\nShape do dataframe limpo:", volunteer_limpo.shape)
