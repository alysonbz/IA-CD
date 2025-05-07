# Importações
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Carregar o dataset
wine = load_wine_dataset()

# Inicializar o scaler
scaler = StandardScaler()

# Excluir a coluna 'Quality' para separar os atributos
X = wine.drop(['Quality'], axis=1)

# Normalizar o dataset
X_norm = scaler.fit_transform(X)

# Obter as labels (rótulos)
y = wine['Quality']

# Print da variância original de X
print('Variância:', X.var())

# Print da variância após normalização
print('Variância do dataset normalizado:', pd.DataFrame(X_norm).var())

# Dividir os dados com estratificação
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, stratify=y, random_state=42)

# Inicializar o KNN
knn = KNeighborsClassifier()

# Treinar o modelo
knn.fit(X_train, y_train)

# Avaliar o modelo
print('Score:', knn.score(X_test, y_test))