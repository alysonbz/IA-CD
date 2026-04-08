# Import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

wine = load_wine_dataset()

# 1. Inicializer o scale
scaler = StandardScaler()
# StandardScaler padroniza os dados para média 0 e desvio padrão 1

# 2. Exclua do dataset a coluna
X = wine.drop(['Quality'], axis=1)
# drop() remove a coluna Quality, deixando apenas as variáveis de entrada

# 3. Normalize o dataset com scaler
X_norm = scaler.fit_transform(X)
# fit_transform() aprende a escala dos dados e já aplica a normalização

# 4. Obtenha as labels da coluna Quality
y = wine['Quality'].values
# values pega os valores da coluna como array (labels do modelo)

# 5. Print a variância de X
print("\nQuestão 5.")
print('variancia', X.var())
# var() calcula a variância de cada coluna do dataset original

# 6. Print a variânca do dataset X_norm
print("\nQuestão 6.")
print('variancia do dataset normalizado', X_norm.var())
# var() aqui mostra que os dados normalizados têm variância próxima de 1

# 7. Divida o dataset em treino e teste com amostragem estratificada
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, stratify=y, random_state=42)
# train_test_split() divide os dados mantendo a proporção das classes (stratify)

# 8. Inicialize o algoritmo KNN
knn = KNeighborsClassifier(n_neighbors=5)
# KNeighborsClassifier cria o modelo KNN com k=5 vizinhos

# 9. Aplique a função fit do KNN
knn.fit(X_train, y_train)
# fit() treina o modelo usando os dados de treino

# 10. Verifique o acerto do classificador
print("\nQuestão 10.")
print('score', knn.score(X_test, y_test))
# score() retorna a acurácia do modelo nos dados de teste