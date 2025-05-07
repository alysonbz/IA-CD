from pandas import value_counts

from src.utils import load_volunteer_dataset
from sklearn.model_selection import train_test_split

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude','Longitude'],axis=1)
print(volunteer_new.info())
print('\n')
# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer_new = volunteer_new.dropna(subset=['category_desc'])
print(volunteer_new)
print('\n')
# mostre o balanceamento das classes em 'category_desc'
print(volunteer_new['category_desc'].value_counts(),'\n','\n')
print('\n')
# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer_new.drop(['category_desc'], axis=1)
print(X)
print('\n')
# Crie um dataframe de labels com a coluna category_desc
y = volunteer_new[['category_desc']]
print(y)
print('\n')
# Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
print(y['category_desc'].value_counts())
print('\n')
print(y_train['category_desc'].value_counts())
print('\n')
print(y_test['category_desc'].value_counts())
print('\n')
# mostre o balanceamento das classes em 'category_desc' novamente
print(volunteer_new['category_desc'].value_counts(), '\n','\n')
