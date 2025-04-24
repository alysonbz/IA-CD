from src.utils import load_volunteer_dataset
from sklearn.model_selection import train_test_split

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer

volunteer_new = volunteer.drop(["Longitude", "Latitude"], axis = 1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new

volunteer = volunteer_new.dropna(subset=["category_desc"])

# mostre o balanceamento das classes em 'category_desc'

print(volunteer['category_desc'].value_counts(),'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``

X = volunteer.drop("category_desc", axis=1)

# Crie um dataframe de labels com a coluna category_desc

y = volunteer[['category_desc']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente

print(y_train["category_desc"].value_counts())


#grafico

import matplotlib.pyplot as plt

class_counts = y_train['category_desc'].value_counts()

plt.figure(figsize=(12, 6))
plt.bar(class_counts.index, class_counts.values, color='skyblue')
plt.title("Distribuiçao das classes y_train", fontsize=16)
plt.xlabel("Categoria", fontsize=12)
plt.ylabel("Contagem", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

