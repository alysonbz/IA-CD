from src.utils import load_volunteer_dataset
from sklearn.model_selection import train_test_split

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(["Latitude", "Longitude"], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=["category_desc"])

# mostre o balanceamento das classes em 'category_desc'
print("Balanceamento antes da divisão:\n", volunteer['category_desc'].value_counts(),'\n\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop("category_desc", axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
print("Balanceamento no conjunto de treino:\n", y_train['category_desc'].value_counts(), '\n')
print("Balanceamento no conjunto de teste:\n", y_test['category_desc'].value_counts())