from fontTools.subset import subset

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(f'Shape do dataset: \n',volunteer.shape)

#mostre os tipos de dados existentes no dataset
print(volunteer.dtypes)

#mostre quantos elementos do dataset estão faltando na coluna
print(f'Total de elementos faltantes por coluna: \n',volunteer.isnull().sum())

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(['Latitude','Longitude'],axis = 1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])

# Print o shape do subset
print(f'Shape do subset: \n',volunteer_subset.shape)


