from src.utils import load_volunteer_dataset
import numpy as np

volunteer = load_volunteer_dataset()

def train_test_split(X,y,test_size,random_seed=1):
    np.random.seed(random_seed)

    n = len(X)
    n_test = int(n* test_size)
    indices = np.random.permutation(n)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train,X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(columns=['Latitude', 'Longitude'])

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer_new = volunteer_new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer_new['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer_new.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer_new['category_desc'].values

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

# mostre o balanceamento das classes em 'category_desc' novamente
print(volunteer_new['category_desc'].value_counts(), '\n', '\n')
