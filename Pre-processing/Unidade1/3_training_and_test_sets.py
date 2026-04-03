from src.utils import load_volunteer_dataset
from sklearn.model_selection import train_test_split

volunteer = load_volunteer_dataset()

# 1. Exclua as colunas Latitude e Longitude de volunteer com a função .drop
volunteer_new = volunteer.drop(columns=['Latitude', 'Longitude'])

# 2. Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer_new = volunteer_new.dropna(subset=['category_desc'])

# 3. Mostre o balanceamento das classes em category_desc utilizando a função .value_counts()
print("\nQuestão 3.")
print(volunteer_new['category_desc'].value_counts())

# 4. Crie um DataFrame com todas as colunas, com exceção de category_desc
X = volunteer_new.drop('category_desc', axis=1)

# 5. Crie um dataframe de labels com a coluna category_desc
y = volunteer_new[['category_desc']]

# 6. Utilize a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 7. Mostre o balanceamento das classes em category_desc novamente
print("\nQuestão 7.")
print(y_train['category_desc'].value_counts())