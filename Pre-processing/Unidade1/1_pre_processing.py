from src.utils import load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
print(f'Na coluna locality tem \033[1;34m{volunteer['locality'].isna().sum()}\033[m valores faltando')

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'], axis=1)
print(volunteer_cols)


# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])
print(volunteer_subset)


# Print o shape do subset
print(volunteer_subset.shape)


