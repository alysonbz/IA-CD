from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


#print(hiking.info())
#print(hiking.head())
#print(wine.describe())
# print(df1.drop([1,2,3]))
# print(df1.drop("A", axis=1))
# print(df1.isna().sum())
# print(df1.dropna(subset=['B']))
#print(df1.dropna(thresh=2))
print(df2.info())

df2["C"] = df2["C"].astype("int64")
print(df2.dtypes)