from sklearn.model_selection import train_test_split

from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer = volunteer.drop(columns=['Latitude', 'Longitude'])

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer_new = volunteer.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer_new.drop(columns=['category_desc'])

# Crie um dataframe de labels com a coluna category_desc
y = volunteer_new[['category_desc']]

# Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
print(volunteer['category_desc'].value_counts(),'\n','\n')


