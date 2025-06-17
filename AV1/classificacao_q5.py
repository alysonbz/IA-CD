import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Carregar dados
df = pd.read_csv('classificacao_ajustado.csv')
X = df.drop(columns=['class_e'])
y = df['class_e']

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Melhor classificador (KNN com StandardScaler e K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# Cross-validation
scores = cross_val_score(knn, X_scaled, y, cv=5)
print(f"Média da acurácia: {scores.mean():.4f}")
print(f"Desvio padrão: {scores.std():.4f}")

# Matriz de confusão e relatório
y_pred = cross_val_predict(knn, X_scaled, y, cv=5)
print("Matriz de Confusão:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
