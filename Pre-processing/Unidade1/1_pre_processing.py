from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# 1. Mostre a dimensão do dataset volunteer
print(volunteer.shape)

# 2. mostre os tipos de dados existentes no dataset
print(volunteer.info())

# 3. mostre quantos elementos do dataset estão faltando na coluna
print(volunteer['locality'].isnull().sum())

# 4. Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(columns=['Latitude', 'Longitude'])

# 5. Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])

# 6. Print o shape do subset
print(volunteer_subset.shape)
