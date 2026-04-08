from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

wine = load_wine_dataset()
scaler = MinMaxScaler()

X = wine.drop(['Quality'], axis=1)
# drop() remove a coluna alvo, deixando apenas as variáveis de entrada

X = np.log(X)
# np.log() aplica log em todas as colunas para reduzir a escala dos dados

y = wine['Quality'].values
# values transforma a coluna em array (labels do modelo)

# 1. Divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# train_test_split() divide os dados mantendo a proporção das classes

knn = KNeighborsClassifier(n_neighbors=3)
# KNeighborsClassifier cria o modelo KNN com 3 vizinhos

# 2. Mostre quantos elementos do dataset estão faltando na coluna Quality
print("\nQuestão 2.")
print(wine['Quality'].isnull().sum())
# isnull().sum() conta quantos valores nulos existem na coluna

# 3. Aplique a função fit do knn
knn.fit(X_train, y_train)
# fit() treina o modelo com os dados de treino

# 4. Mostre o acerto do algoritmo
print("\nQuestão 4.")
print(knn.score(X_test, y_test))
# score() retorna a acurácia do modelo nos dados de teste