import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import load_fish_dataset

# Carrega os dados
df = load_fish_dataset()
X = df.drop(['specie'], axis=1)
y = df['specie']

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA para reduzir para 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

# Modelo de classificação (KNN)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

# Previsões
y_pred = clf.predict(X_test)

# Avaliação
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))