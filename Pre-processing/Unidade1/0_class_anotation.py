from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
from sklearn.model_selection import train_test_split
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

print("\n",df2.info())
print(df2)

X = volunteer.drop('Category_desc', axis=1)
y = volunteer['Category_desc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Conjunto de iformações para treino: ", y_train.value_counts())

print("\nConjunto de iformações para treino: ", y_test.value_counts())