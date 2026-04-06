from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# 1. Mostre a dimensão do dataset volunteer usando a função .shape
print("\nQuestão 1.")
print(volunteer.shape)

# 2. Mostre os tipos de dados existentes no dataset usando a função .info()
print("\nQuestão 2.")
volunteer.info()

# 3. Mostre quantos elementos do dataset estão faltando na coluna locality
print("\nQuestão 3.")
print(volunteer['locality'].isnull().sum())

# 4. Exclua as colunas Latitude e Longitude de volunteer e coloque em um dataframe volunteer_cols
volunteer_cols = volunteer.drop(columns=['Latitude', 'Longitude'])

# 5. Exclua as linhas com valores null da coluna category_desc de volunteer_cols e coloque em um dataframe volunteer_subset
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])

# 6. Print o shape de volunteer_subset
print("\nQuestão 6.")
print(volunteer_subset.shape)