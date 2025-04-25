from src.utils import load_volunteer_dataset
import numpy as np
volunteer = load_volunteer_dataset()

def train_test_split(X,y,df_stratify,test_size,random_seed=1):
    np.random.seed(random_seed)

    train_idx = []
    test_idx = []

    for classe in y['category_desc'].unique():
        #embaralhando por classe
        indices_classe = y[y['category_desc'] == classe].index.to_numpy()
        np.random.shuffle(indices_classe)

        #definindo o tamanho de amostra da classe especifica em total e em test
        n_total = len(indices_classe)
        n_test = int(np.floor(test_size * n_total))

        train_idx.extend(indices_classe[:n_test])
        test_idx.extend(indices_classe[n_test:])


    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]


    return X_train,X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude','Longitude'],axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset='category_desc')

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop(['category_desc'], axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X,y,y,test_size,random_seed=1)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train['category_desc'].value_counts(normalize=True).round(2),y_test['category_desc'].value_counts(normalize=True).round(2))