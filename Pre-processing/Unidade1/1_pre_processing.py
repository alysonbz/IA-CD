from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
print(volunteer.dtypes)

#mostre quantos elementos do dataset estão faltando na coluna
volunteer['category_desc'].isnull().sum()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(columns=['Latitude', 'Longitude'])

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
#volunteer_subset =

# Print o shape do subset
#print(volunteer_subset.shape)


