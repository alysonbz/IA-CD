from src.utils import load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
<<<<<<< HEAD
# print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna locality
# print(volunteer["category_desc"].isnull().sum())

# Exclua as colunas Latitude e Longitude de volunteer
# volunteer_cols = volunteer.drop(["Latitude", "Longitude"], axis=1)
=======
#print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols
>>>>>>> 360b412b54f5b63b0a5f4a0a712e59529a54e83b

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
# volunteer_subset = volunteer.dropna(subset=["category_desc"])

# Print o shape do subset
<<<<<<< HEAD
# print(volunteer_subset.shape)
#
=======


>>>>>>> 360b412b54f5b63b0a5f4a0a712e59529a54e83b

