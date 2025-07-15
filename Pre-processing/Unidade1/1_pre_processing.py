from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print('shape \n')
print(volunteer.shape)

print('isnull\n')
print(volunteer['locality'].isnull().sum())

#mostre os tipos de dados existentes no dataset
print('dtypes \n')
print(volunteer.dtypes)

#mostre quantos elementos do dataset estão faltando na coluna
print('descrevendo \n')
print(volunteer.isna().sum())

# Exclua as colunas Latitude e Longitude de volunteer
print('volunteer_cols')
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'] ,axis=1)
print(volunteer_cols.info())

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer.dropna(subset = ['category_desc'])

# Print o shape do subset
print('sub set')
print(volunteer_subset.shape)


