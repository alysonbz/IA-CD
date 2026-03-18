from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


print(wine.describe())


print(df1.dropna())
print('-'*10)

print(df1.drop(['A'], axis=1))
print('-'*10)

print(df1.drop([1,2,3], axis=0))
print('-'*10)

print(df1.dropna(thresh=2))
print('-'*10)

df1.dropna(subset=['A'], inplace=True)
df1['A'] = df1['A'].astype('int64')
df1.info()
