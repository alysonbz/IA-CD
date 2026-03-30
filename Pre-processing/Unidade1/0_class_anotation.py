from sklearn.model_selection import train_test_split
from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()

print(volunteer.head())

X = volunteer.drop('category_desc', axis = 1)
y = volunteer['category_desc']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("Conjunto de informações para treino:", y_train.value_counts())