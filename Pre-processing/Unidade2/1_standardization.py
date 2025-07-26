from src.utils import load_wine_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Carregar o dataset
wine = load_wine_dataset()

# Separar variáveis independentes e alvo
X = wine.drop(['Quality'], axis=1)
y = wine['Quality'].values

# Divida o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Instanciar o KNN
knn = KNeighborsClassifier()

# Treinar o modelo
knn.fit(X_train, y_train)

# Mostrar a acurácia do algoritmo no conjunto de teste
print(knn.score(X_test, y_test))