from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


print(wine.info()) #informações das colunas

print(df1.drop([1, 2, 3])) #excluir linha

print(df1.drop("C", axis=1)) #excluir coluna

print(df1.isna().sum()) #somar valores ausentes de cada coluna

print(df1.dropna(subset=['A'])) #excluir linhas com valores ausentes

print(df1.dropna(thresh=2)) #excluir as linhas com 2 NaN

print(df2)
print(df2.info())