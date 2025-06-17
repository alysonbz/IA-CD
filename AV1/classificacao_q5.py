import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

# 1. Carregar e preparar os dados
df = pd.read_csv("classificacao_ajustado.csv")

# 2. Selecionar colunas numéricas úteis
X = df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift']]
y = df['class']

# 2. Melhor configuração encontrada: RobustScaler + k=3
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=3)


# 3. Cross-validation (5 folds)
scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
print(f"\nMédia da Acurácia (5 folds): {scores.mean():.4f}")
print(f"Desvio Padrão da Acurácia: {scores.std():.4f}")

# 4. Previsões para Matriz de Confusão e Classification Report
y_pred = cross_val_predict(knn, X_scaled, y, cv=5)

cm = confusion_matrix(y, y_pred)
print("\nMatriz de Confusão:")
print(cm)

print("\nRelatório de Classificação:")
print(classification_report(y, y_pred))

# O modelo KNN (k=3) com RobustScaler obteve 95% de acurácia média em 5 folds.
# Apresentou ótimo desempenho nas três classes, com f1-scores entre 0.93 e 0.95.
# Mesmo com desbalanceamento, manteve boa precisão e recall para todas as classes.
