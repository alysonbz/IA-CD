from src.utils import load_volunteer_dataset
import numpy as np

volunteer = load_volunteer_dataset()

def train_test_split(X,y,test_size,random_seed=1):
    #SEU CÓDIGO AQUI
    np.random.seed(random_seed)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_len = int(len(X) * test_size)

    indices_teste = indices[:test_len]
    indices_treino = indices[test_len:]

    X_train = X.iloc[indices_treino]
    X_test = X.iloc[indices_teste]
    y_train = y.iloc[indices_treino]
    y_test = y.iloc[indices_teste]

    return X_train,X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(["Latitude", "Longitude"], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer_new = volunteer_new.dropna(subset=["category_desc"])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop("category_desc", axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[["category_desc"]]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size,random_seed=1)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train['category_desc'].value_counts(), '\n')
print(y_test['category_desc'].value_counts())