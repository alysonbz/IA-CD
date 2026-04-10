from src.utils import load_volunteer_dataset
import pandas as pd
import random

volunteer = load_volunteer_dataset()

def train_test_split(X,y,test_size,random_seed=1):
    random.seed(random_seed)
    n = len(X)

    indices = list(range(n))    # Cria índices e embaralha
    random.shuffle(indices)

    n_test = int(n * test_size)     # Calcular tamanho do teste

    test_idx = indices[:n_test]     # Separa índices
    train_idx = indices[n_test:]

    # Criar conjuntos
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    return X_train,X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer['category_desc']

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size,random_seed=1)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train.value_counts())
print(y_test.value_counts())