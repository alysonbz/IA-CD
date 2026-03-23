from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print("A dimensão do dataset é: ", volunteer.shape)  # O atributo .shape retorna uma tupla contendo o número de linhas e colunas.

#mostre os tipos de dados existentes no dataset
print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
print('\n',volunteer['locality'].isnull().sum())   # .isnull().sum() para contar quantos campos estão vazios em uma coluna específica.

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'], axis=1)


# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols[volunteer_cols['category_desc'].notnull()]

# Print o shape do subset
print("\nNova dimensão após a limpeza: ", volunteer_subset.shape)


