from itertools import count
from random import shuffle
import random
import pandas as pd
from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()


def train_test_split(X, y, test_size, random_seed=1):
    X2 = pd.DataFrame()
    y2 = pd.DataFrame()
    t = len(X)
    # numeros_unicos = random.sample(range(0, 616), 617)
    # for i in range (t):
    #    X2.loc[i] = X.loc[numeros_unicos[i]]
    teste = int(t*test_size)
    X_train = X[teste:]
    X_test = X[:teste]
    y_train = y[teste:]
    y_test = y[:teste]

    return X_train, X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(["Latitude", "Longitude"], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=["category_desc"])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n', '\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop(['category_desc'], axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer['category_desc']

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size, random_seed=1)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train.value_counts(), '\n', '\n')
print(y_test.value_counts(), '\n', '\n')
