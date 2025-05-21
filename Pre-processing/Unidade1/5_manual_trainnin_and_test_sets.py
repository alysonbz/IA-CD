from src.utils import load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()

# Função personalizada para dividir treino e teste com estratificação
def train_test_split(X, y, test_size, random_seed=1):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_seed)

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n', '\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# Utiliza a amostragem estratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_seed=1)

# mostre o balanceamento das classes em 'category_desc' novamente (após split)
print(y_train['category_desc'].value_counts(normalize=True))
print(y_test['category_desc'].value_counts(normalize=True))