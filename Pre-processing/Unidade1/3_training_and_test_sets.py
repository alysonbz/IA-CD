from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer = volunteer.drop(columns=["Latitude", "Longitude"])

 # Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer.dropna(subset=["category_desc"])


# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts())

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop(columns=["category_desc"])

# # Crie um dataframe de labels com a coluna category_desc
y =  volunteer[["category_desc"]]

# Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = __(__, __, stratify=__, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
print("Distribuição em y_train:")
print(y_train['category_desc'].value_counts(normalize=True))