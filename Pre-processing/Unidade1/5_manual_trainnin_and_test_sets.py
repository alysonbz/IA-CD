from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

def train_test_split(X,y,test_size,random_seed=1):
    #SEU CÓDIGO AQUI
    return X_train,X_test, y_train, y_test

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude','Longitude'], axis=1)
print(volunteer_new)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer_new = volunteer_new.dropna(subset=['category_desc'])
print(volunteer_new)

# mostre o balanceamento das classes em 'category_desc'
print(volunteer_new['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop(['category_desc'], axis=1)
print(X)
# Crie um dataframe de labels com a coluna category_desc
y = volunteer_new[['category_desc']]
print(y)
# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = volunteer_new[['category_desc']]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size,random_seed=1)
print(y['category_desc'].value_counts())
print('\n')
print(y_train['category_desc'].value_counts())
print('\n')
print(y_test['category_desc'].value_counts())
print('\n')

# mostre o balanceamento das classes em 'category_desc' novamente
print(volunteer_new['category_desc'].value_counts(),'\n','\n')
