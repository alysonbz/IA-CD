from src.utils import load_volunteer_dataset
import pandas as pd
import os

caminho_csv = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'opportunities.csv')
caminho_csv = os.path.abspath(caminho_csv)
volunteer = pd.read_csv(caminho_csv)

def train_test_split(X,y,test_size,random_seed=1):
    #SEU CÓDIGO AQUI
    dados = X.join(y)

    partes_treino = []
    partes_teste = []

    for classe in y['category_desc'].unique():
        dados_classe = dados[dados['category_desc'] == classe]

        teste_classe = dados_classe.sample(frac=test_size, random_state=random_seed)
        treino_classe = dados_classe.drop(teste_classe.index)

        partes_treino.append(treino_classe)
        partes_teste.append(teste_classe)

    treino = pd.concat(partes_treino).sample(frac=1, random_state=random_seed)
    teste = pd.concat(partes_teste).sample(frac=1, random_state=random_seed)

    X_train = treino.drop('category_desc', axis=1)
    X_test = teste.drop('category_desc', axis=1)

    y_train = treino[['category_desc']]
    y_test = teste[['category_desc']]

    return X_train,X_test, y_train, y_test

# Exclua as colunas Latitude e Longitude de volunteer
print(volunteer.columns)
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n', '\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size,random_seed=1)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train['category_desc'].value_counts(), '\n')
print(y_test['category_desc'].value_counts())