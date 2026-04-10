from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

## realize print do dataset volunteer corrigido sem nenhum NAN, para isto removam as colunas NAN e depois as linhas e crie
#um dataframe novo e print este mostrando a contagem de colunas NAN existentes e mostre também o shape novo.

import pandas as pd
from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Remove colunas onde TODOS os valores são NaN
volunteer_clean = volunteer.dropna(axis=1, how='all')

# Remove colunas com mais de 50% de NaN
threshold = len(volunteer_clean) * 0.5
volunteer_clean = volunteer_clean.dropna(axis=1, thresh=int(threshold))

# Remove linhas com qualquer NaN restante
volunteer_clean = volunteer_clean.dropna(axis=0, how='any')

# Print contagem de NaN (deve ser 0 em tudo)
print("Contagem de NaN por coluna:")
print(volunteer_clean.isnull().sum(), '\n')

# Print shape novo
print("Shape novo:", volunteer_clean.shape)