from pandas import isnull

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
print(f'O total de elementos faltantes em Locality é: {volunteer['locality'].isna().sum()}')

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(columns=['Latitude', 'Longitude'])
print(volunteer_cols.info())


# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])

# Print o shape do subset
print(volunteer_subset.shape)

print(volunteer_cols['category_desc'].isnull().sum())
print(volunteer_subset['category_desc'].isnull().sum())





