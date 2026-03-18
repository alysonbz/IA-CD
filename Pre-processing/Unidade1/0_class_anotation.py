from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


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
