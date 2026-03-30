from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

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


X = volunteer.drop('category_desc', axis=1)
y = volunteer[['category_desc']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

print(X_train)
print(y_train)