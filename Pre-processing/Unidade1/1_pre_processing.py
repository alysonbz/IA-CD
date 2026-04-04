from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print("\n\nShape: ",volunteer.shape)


#mostre os tipos de dados existentes no dataset
print(volunteer.info())

#mostre quantos elementos do dataset estão faltando na coluna
print("\n",volunteer["locality"].isna().sum())

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_cols = volunteer.drop(["Latitude","Longitude"],axis=1)
print(volunteer_cols)

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=["category_desc"])
print("\n\nShape: ",volunteer_subset.shape)

# Print o shape do subset

print(wine.describe())
print(wine.info())
print(df1)
print("\n",df1.dropna())
print("\n",df1.drop([1,2,4]))
print("\n",df1.isna().sum())
print("\n",df1.dropna(subset=["B"]))
print("\n",df1.dropna(thresh=2))


print("\n\nShape: ",volunteer.shape)

print("\n",volunteer.info())

print("\n",volunteer["locality"].isna().sum())

volunteer_cols = volunteer.drop(["Latitude","Longitude"],axis=1)
print(volunteer_cols)

volunteer_subset = volunteer_cols.dropna(subset=["category_desc"])
print("\n\nShape: ",volunteer_subset.shape)

