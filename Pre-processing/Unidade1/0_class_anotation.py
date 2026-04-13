from sklearn.model_selection import train_test_split

from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


print(wine.describe())


print(df1.drop([1, 2, 3]))

print(df1.drop("A", axis=1))

print(df1.isna().sum())

print(df1.dropna(subset=["B"]))

print(df1.dropna(thresh=2))

print(df2.info())

x = volunteer.drop('category_desc', axis=1)
y = volunteer['category_desc']

x_train, x_test, y_train, y_test = train_test_split(  x, y, test_size=0.2, random_state=42)
