from src.utils import load_volunteer_dataset
import pandas as pd
import numpy as np

volunteer = load_volunteer_dataset()

def train_test_split(X, y, test_size, random_seed=1):
    # Definindo a semente para reprodutibilidade
    np.random.seed(random_seed)

    # Criando um índice aleatório
    shuffled_indices = np.random.permutation(len(X))

    # Calculando o tamanho do conjunto de teste
    test_set_size = int(len(X) * test_size)

    # Separando os índices para teste e treino
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    # Criando os conjuntos finais usando .iloc para garantir a seleção correta por posição
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    return X_train,X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(["Latitude", "Longitude"], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer_new['category_desc'].value_counts,'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size,random_seed=1)

# mostre o balanceamento das classes em 'category_desc' novamente
print("Balanciamento do Treino:\n", y_train['category_desc'].value_counts())
print("Balanciamento no Teste:\n", y_test['category_desc'].value_counts())