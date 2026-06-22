from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import load_fish_dataset

# Carrega o dataset
df = load_fish_dataset()

# Separa atributos e classe
X = df.drop('specie', axis=1)
y = df['specie']

# Transforma as espécies em números
le = LabelEncoder()
y = le.fit_transform(y)

# Padroniza os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# PCA com 2 componentes
pca = PCA(n_components=2)
X = pca.fit_transform(X)

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

# Cria o classificador
knn = KNeighborsClassifier()

# Treina o modelo
knn.fit(X_train, y_train)

# Faz as previsões
y_pred = knn.predict(X_test)

# Resultados
print("Acurácia:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))